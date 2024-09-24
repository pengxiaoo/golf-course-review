import os
import time
import datetime
import pandas as pd
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
import csv

# refer to
# https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?tabs=standard-input&pivots=programming-language-python

load_dotenv()
# use constants defined in .env file
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

text_col_num = 5
encoding_used = 'utf-8'
batch_endpoint = "/chat/completions"
input_data = "input_data/golf_course_reviews.csv"
output_data = "output_data/result.jsonl"
output_result = "output_data/result.csv"
final_output_result = "output_data/final_result.csv"


def convert_csv_to_jsonl(csv_file_path: str, prompt: str) -> str:
    """
    :param csv_file_path: path of the input csv file
    :param prompt: prompt used in chat/completions
    :return: path of the generated jsonl file
    """
    try:
        df = pd.read_csv(csv_file_path, encoding=encoding_used)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file_path, encoding='windows-1252')
        except UnicodeDecodeError as e:
            print(f"Failed to read CSV file: {e}")

    text_col = df.columns[text_col_num - 1]
    jsonl_output = []
    for idx, row in df.iterrows():
        # Prepare the main text
        json_obj = {"custom_id": str(idx), "method": "POST", "url": f"{batch_endpoint}"}
        json_obj["body"] = \
            {
                "model": "course-review",
                "messages": [
                    {"role": "system", "content": "You are a sentiment analysis model."},
                    {"role": "user", "content": f"{prompt} {str(row[text_col])}"}
                ]
            }
        # Serialize the JSON object to a string
        json_str = json.dumps(json_obj)
        jsonl_output.append(json_str)

    # Write to JSONL file
    output_file_path = csv_file_path.replace('.csv', '.jsonl')
    with open(output_file_path, 'w') as file:
        for item in jsonl_output:
            file.write(item + '\n')
    return output_file_path


def upload_file(csv_file_path: str, prompt: str) -> str:
    jsonl_file_path = convert_csv_to_jsonl(csv_file_path, prompt)
    # Upload a file with a purpose of "batch"
    file = client.files.create(
        file=open(jsonl_file_path, "rb"),
        purpose="batch"
    )
    return file.id


def create_batch_job(file_id: str) -> str:
    # Create a batch job
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h",
    )
    print(batch_response.model_dump_json(indent=2))
    return batch_response.id


def upload_and_create_job(csv_file_path: str, prompt: str) -> str:
    file_id = upload_file(csv_file_path, prompt)
    status = "pending"
    while status != "processed":
        time.sleep(15)
        file_response = client.files.retrieve(file_id)
        status = file_response.status
        print(f"{datetime.datetime.now()} File Id: {file_id}, Status: {status}")
    return create_batch_job(file_id)


def track_and_save_job_result(batch_id: str) -> str:
    batch_response = None
    status = "validating"
    output_file_id = None
    result_file_path = None
    while status not in ("completed", "failed", "canceled"):
        time.sleep(60)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        output_file_id = batch_response.output_file_id
        print(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

    if status == "failed":
        if batch_response:
            for error in batch_response.errors.data:
                print(f"Error code {error.code} Message {error.message}")
    elif status == "completed" and output_file_id:
        print(f"Output file: {output_file_id}")
        result_file_path = save_job_result(output_file_id)
    else:
        print("The batch job is canceled.")
    # result_file_path = save_job_result("file-a5f8ea0c-ad08-4303-93b0-b6f74251ec53")
    return result_file_path


def save_job_result(output_file_id) -> str:
    file_response = client.files.content(output_file_id)
    raw_responses = file_response.text.strip().split('\n')
    result_file_path = output_result
    # save the result to a file in /output_data
    json_data = []
    for raw_response in raw_responses:
        json_response = json.loads(raw_response)  # Convert raw JSON string to Python dict
        json_data.append(json_response)  # Add each JSON object to the list

    if json_data:
        # Get the header from the keys of the first JSON object
        csv_file = output_result
        header = ['id', 'sentiment']
        # Open the CSV file for writing
        with open(csv_file, 'w', newline='', encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            # Write the header to the CSV file
            writer.writerow(header)
            # Write each row (JSON data)
            json_data = sorted(json_data, key=lambda x: x['custom_id'])
            for entry in json_data:
                custom_id = str(int(entry.get("custom_id")) + 1)
                content = entry['response']['body']['choices'][0]['message']['content'].capitalize()
                try:
                    writer.writerow([custom_id, content])
                except Exception as e:
                    print(custom_id, "=========", content)
                    writer.writerow([custom_id, "Neutral"])
        print(f"Data has been exported to {csv_file}")
    else:
        print("No data to export to CSV.")
    return result_file_path


def join_results_with_original_data(csv_file_path: str, result_file_path: str):
    df1 = pd.read_csv(result_file_path)
    df2 = pd.read_csv(csv_file_path)
    df2['id'] = range(1, len(df2) + 1)
    columns = ['id'] + [col for col in df2.columns if col != 'id']
    df2 = df2[columns]

    df3 = pd.merge(df2, df1, on="id")
    df = df3.drop(['id'], axis=1)
    df.to_csv(final_output_result, encoding='utf-8', index=False, quoting=csv.QUOTE_ALL)
    pass


if __name__ == "__main__":
    # todo: use sample data first, after testing, we can change to full data
    csv_file_path = input_data
    prompt_templates = [
        """
            Please classify the sentiment of this text as either 'positive' or 'negative' 
            or 'neutral'. Translate non-English content before answering. Respond with one word only.
        """,
    ]
    for prompt_template in prompt_templates:
        batch_id = upload_and_create_job(csv_file_path, prompt_template)
        result_file_path = track_and_save_job_result(batch_id)
        if result_file_path:
            join_results_with_original_data(csv_file_path, result_file_path)
