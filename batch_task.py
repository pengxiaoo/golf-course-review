import os
from enum import Enum
import time
import datetime
import pandas as pd
import json
import csv
from dotenv import load_dotenv
from openai import AzureOpenAI


class BatchTaskType(str, Enum):
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"


class BatchTask:

    def __init__(self,
                 task_type: BatchTaskType,
                 input_data_path,
                 model="course-review",
                 batch_endpoint="/chat/completions",
                 encoding_used="utf-8",
                 comment_col_name="comment",
                 ):
        self.task_type = task_type
        self.input_data_path = input_data_path
        self.concatenated_data_path = input_data_path.replace(".csv", "_concatenated.csv")
        time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.llm_result_data_path = f"output_data/llm_result_{task_type.value}_{time_str}.csv"
        self.join_result_data_path = f"output_data/join_result_{task_type.value}_{time_str}.csv"
        self.model = model
        self.batch_endpoint = batch_endpoint
        self.encoding_used = encoding_used
        self.comment_col_name = comment_col_name
        self.batch_id = None
        self.head_number = 20
        # use constants defined in .env file
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("API_KEY"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def get_prompt(self) -> str:
        if self.task_type == BatchTaskType.SENTIMENT:
            return """
                The following is a review of a golf course from a golfer who had played there.
                Please classify the sentiment of this review as either "positive" or "negative" or "neutral". 
                Translate non-English content before answering, ignore unicodes and unrecognized words. 
                Respond with one word only.
                The review text is:
            """
        elif self.task_type == BatchTaskType.SUMMARIZATION:
            return """
                The following are reviews of a golf course from multiple golfers who had played there recently.
                individual reviews are started with a new line.
                Please summarize the reviews into a single paragraph, in 40~80 words, 
                so that new golfers can quickly know about the golf course. 
                Translate non-English content before answering, ignore unicodes and unrecognized words.
                The text of the reviews is:
            """
        elif self.task_type == BatchTaskType.EXTRACTION:
            return """
                The following are reviews of a golf course from multiple golfers who had played there recently.
                individual reviews are started with a new line.
                Please extract attribute the reviews in 3 words, 
                so that new golfers can quickly know about the golf course. 
                Translate non-English content before answering, ignore unicodes and unrecognized words.
                The text of the reviews is:
            """
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def run(self):
        # see https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch?pivots=programming-language-python
        self.batch_id = self.upload_and_create_job()
        if self.batch_id is not None:
            llm_success = self.track_and_save_job_result()
            if llm_success and self.task_type == BatchTaskType.SENTIMENT:
                self.join_results_with_original_data()
            elif self.task_type == BatchTaskType.SUMMARIZATION:
                self.join_result_with_concatenated_data()
            elif self.task_type == BatchTaskType.EXTRACTION:
                self.join_result_with_concatenated_data()

    def upload_and_create_job(self):
        file_id = self.upload_file()
        if file_id is None:
            return None
        status = "pending"
        while status != "processed":
            time.sleep(15)
            file_response = self.client.files.retrieve(file_id)
            status = file_response.status
            print(f"{datetime.datetime.now()} File Id: {file_id}, Status: {status}")
        return self.create_batch_job(file_id)

    def get_dataframe_summarization(self):
        csv_file_path = self.input_data_path
        df = pd.read_csv(csv_file_path)
        df_sorted = df.sort_values(by=["comment_time"], ascending=False)
        requests = (df_sorted.groupby(["course_id", "club_id"])["comment"]
                    .apply(lambda x: "\n".join(x.head(self.head_number).fillna('')))
                    .reset_index())
        requests_df = pd.DataFrame(requests)
        requests_df.to_csv(self.concatenated_data_path, index=False, quoting=csv.QUOTE_ALL)
        return requests_df

    def upload_file(self):
        jsonl_file_path = self.convert_csv_to_jsonl()
        if jsonl_file_path is None:
            return None
        file = self.client.files.create(
            file=open(jsonl_file_path, "rb"),
            purpose="batch"
        )
        return file.id

    def convert_csv_to_jsonl(self):
        csv_file_path = self.input_data_path
        try:
            if self.task_type == BatchTaskType.SUMMARIZATION or self.task_type == BatchTaskType.EXTRACTION:
                df = self.get_dataframe_summarization()
            else:
                df = pd.read_csv(csv_file_path, encoding=self.encoding_used)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file_path, encoding="windows-1252")
            except UnicodeDecodeError as e:
                print(f"Failed to read CSV file: {e}")
                return None

        prompt = self.get_prompt()
        jsonl_output = []
        for idx, row in df.iterrows():
            # Prepare the main text
            json_obj = {
                "custom_id": str(idx),  # custom_id is the row number(starts from 0) of the input csv file
                "method": "POST",
                "url": self.batch_endpoint,
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": f"{prompt} {str(row[self.comment_col_name])}"}
                    ]
                }
            }
            # Serialize the JSON object to a string
            json_str = json.dumps(json_obj)
            jsonl_output.append(json_str)

        # Write to JSONL file
        output_file_path = f"input_data/requests_{self.task_type.value}.jsonl"
        with open(output_file_path, "w") as file:
            for item in jsonl_output:
                file.write(item + "\n")
        return output_file_path

    def create_batch_job(self, file_id: str) -> str:
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint=self.batch_endpoint,
            completion_window="24h",
        )
        print(batch_response.model_dump_json(indent=2))
        return batch_response.id

    def track_and_save_job_result(self) -> bool:
        batch_response = None
        status = "validating"
        output_file_id = None
        while status not in ("completed", "failed", "canceled"):
            time.sleep(60)
            batch_response = self.client.batches.retrieve(self.batch_id)
            status = batch_response.status
            output_file_id = batch_response.output_file_id
            print(f"{datetime.datetime.now()} Batch Id: {self.batch_id},  Status: {status}")

        if status == "failed":
            if batch_response:
                for error in batch_response.errors.data:
                    print(f"Error code {error.code} Message {error.message}")
        elif status == "completed" and output_file_id:
            print(f"Output file: {output_file_id}")
            return self.save_job_result(output_file_id)
        else:
            print("The batch job is canceled.")
        return False

    def save_job_result(self, output_file_id) -> bool:
        file_response = self.client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split("\n")
        json_data = []
        for raw_response in raw_responses:
            json_response = json.loads(raw_response)  # Convert raw JSON string to Python dict
            json_data.append(json_response)
        if len(json_data) > 0:
            # Get the header from the keys of the first JSON object
            header = ["custom_id", "result"]
            # Open the CSV file for writing
            with open(self.llm_result_data_path, "w", newline="", encoding=self.encoding_used,
                      errors="ignore") as csv_file:
                writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                # Write the header to the CSV file
                writer.writerow(header)
                # Write each row (JSON data)
                json_data = sorted(json_data, key=lambda x: int(x["custom_id"]))
                for entry in json_data:
                    custom_id = entry.get("custom_id")
                    try:
                        result = entry["response"]["body"]["choices"][0]["message"]["content"].capitalize()
                        writer.writerow([custom_id, result])
                    except Exception as e:
                        print(f"Error in save_job_result: {e}")
                        print(custom_id, "=========", result)
                        writer.writerow([custom_id, f"${self.task_type} result not available"])
            print(f"Data has been exported to {self.llm_result_data_path}")
            return True
        else:
            print("No data to export to CSV.")
            return False

    def join_results_with_original_data(self):
        df_input_data = pd.read_csv(self.input_data_path)
        df_input_data["custom_id"] = range(0, len(df_input_data))
        df_llm_result = pd.read_csv(self.llm_result_data_path)
        df_merged = pd.merge(df_input_data, df_llm_result, on="custom_id") \
            .drop(["custom_id"], axis=1)
        df_merged.to_csv(self.join_result_data_path, encoding=self.encoding_used, index=False, quoting=csv.QUOTE_ALL)

    def join_result_with_concatenated_data(self):
        df_concatenated_reviews = pd.read_csv(self.concatenated_data_path)
        df_concatenated_reviews["custom_id"] = range(0, len(df_concatenated_reviews))
        df_llm_result = pd.read_csv(self.llm_result_data_path)
        df_merged = pd.merge(df_concatenated_reviews, df_llm_result, on="custom_id") \
            .drop(["custom_id"], axis=1)
        df_merged.to_csv(self.join_result_data_path, encoding=self.encoding_used, index=False, quoting=csv.QUOTE_ALL)
