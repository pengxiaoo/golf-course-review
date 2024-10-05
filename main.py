from batch_task import BatchTask, BatchTaskType

if __name__ == "__main__":
    extraction_task = BatchTask(
        task_type=BatchTaskType.EXTRACTION,
        input_data_path="input_data/golf_course_reviews.csv",
    )
    extraction_task.run()
