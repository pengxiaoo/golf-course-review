from batch_task import BatchTask, BatchTaskType

if __name__ == "__main__":

    summarization_task = BatchTask(
        task_type=BatchTaskType.SUMMARIZATION,
        input_data_path="input_data/golf_course_reviews.csv",
    )
    summarization_task.run()
