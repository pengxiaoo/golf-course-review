import os
import time
import datetime
import pandas as pd
import json
import csv
from dotenv import load_dotenv
from openai import AzureOpenAI

# refer to
# https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?tabs=standard-input&pivots=programming-language-python

load_dotenv()
# use constants defined in .env file
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

use_sample_data = False
comment_col_name = "comment"
encoding_used = "utf-8"
model = "course-review"
batch_endpoint = "/chat/completions"
input_data = "input_data/golf_course_reviews_sample.csv" if use_sample_data \
    else "input_data/golf_course_reviews.csv"
llm_output_result = "output_data/llm_result.csv"
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
            df = pd.read_csv(csv_file_path, encoding="windows-1252")
        except UnicodeDecodeError as e:
            print(f"Failed to read CSV file: {e}")

    jsonl_output = []
    for idx, row in df.iterrows():
        # Prepare the main text
        json_obj = {
            "custom_id": str(idx),  # custom_id is the row number(starts from 0) of the input csv file
            "method": "POST",
            "url": batch_endpoint,
            "body": {
                "model": model,
                "messages": [
                    {"role": "user", "content": f"{prompt} {str(row[comment_col_name])}"}
                ]
            }
        }
        # Serialize the JSON object to a string
        json_str = json.dumps(json_obj)
        jsonl_output.append(json_str)

    # Write to JSONL file
    output_file_path = csv_file_path.replace(".csv", ".jsonl")
    with open(output_file_path, "w") as file:
        for item in jsonl_output:
            file.write(item + "\n")
    return output_file_path


def upload_file(csv_file_path: str, prompt: str) -> str:
    jsonl_file_path = convert_csv_to_jsonl(csv_file_path, prompt)
    file = client.files.create(
        file=open(jsonl_file_path, "rb"),
        purpose="batch"
    )
    return file.id


def create_batch_job(file_id: str) -> str:
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint=batch_endpoint,
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


def track_and_save_job_result(batch_id: str):
    batch_response = None
    status = "validating"
    output_file_id = None
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
        return save_job_result(output_file_id)
    else:
        print("The batch job is canceled.")
    return None


def save_job_result(output_file_id):
    file_response = client.files.content(output_file_id)
    raw_responses = file_response.text.strip().split("\n")
    # save the result to a file in /output_data
    json_data = []
    for raw_response in raw_responses:
        json_response = json.loads(raw_response)  # Convert raw JSON string to Python dict
        json_data.append(json_response)
    if len(json_data) > 0:
        # Get the header from the keys of the first JSON object
        header = ["custom_id", "result"]
        # Open the CSV file for writing
        with open(llm_output_result, "w", newline="", encoding=encoding_used) as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            # Write the header to the CSV file
            writer.writerow(header)
            # Write each row (JSON data)
            json_data = sorted(json_data, key=lambda x: int(x["custom_id"]))
            for entry in json_data:
                custom_id = entry.get("custom_id")
                sentiment = entry["response"]["body"]["choices"][0]["message"]["content"].capitalize()
                try:
                    writer.writerow([custom_id, sentiment])
                except Exception as e:
                    print(f"Error in save_job_result: {e}")
                    print(custom_id, "=========", sentiment)
                    writer.writerow([custom_id, "Neutral"])
        print(f"Data has been exported to {llm_output_result}")
        return llm_output_result
    else:
        print("No data to export to CSV.")
        return None


def join_results_with_original_data(input_csv_path, llm_result_file_path):
    df_input_data = pd.read_csv(input_csv_path)
    df_input_data["custom_id"] = range(0, len(df_input_data))
    df_llm_result = pd.read_csv(llm_result_file_path)
    df_merged = pd.merge(df_input_data, df_llm_result, on="custom_id") \
        .drop(["custom_id"], axis=1)
    df_merged.to_csv(final_output_result, encoding=encoding_used, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    prompt_templates = [
        """
            Please classify the sentiment of this text as either "positive" or "negative" or "neutral". 
            Translate non-English content before answering. Respond with one word only.
            The text is:
        """,
    ]
    for prompt_template in prompt_templates:
        batch_id = upload_and_create_job(input_data, prompt_template)
        llm_result_file_path = track_and_save_job_result(batch_id)
        if llm_result_file_path:
            join_results_with_original_data(input_data, llm_result_file_path)
