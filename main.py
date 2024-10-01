from batch_task import BatchTask, BatchTaskType

if __name__ == "__main__":
<<<<<<< HEAD
    # sentiment_task = BatchTask(
    #     task_type=BatchTaskType.SENTIMENT,
    #     input_data_path="input_data/golf_course_reviews.csv",
    # )
    # sentiment_task.run()

    summarization_task = BatchTask(
=======
    sentiment_task = BatchTask(
>>>>>>> dabcb36 (minor updates)
        task_type=BatchTaskType.SUMMARIZATION,
        input_data_path="input_data/golf_course_reviews.csv",
    )
    summarization_task.run()
