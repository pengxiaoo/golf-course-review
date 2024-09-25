from batch_task import BatchTask, BatchTaskType

if __name__ == "__main__":
    sentiment_task = BatchTask(
        task_type=BatchTaskType.SENTIMENT,
        input_data_path="input_data/golf_course_reviews.csv",
    )
    sentiment_task.run()
