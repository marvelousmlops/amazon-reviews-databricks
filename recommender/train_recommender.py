from pyspark.sql import SparkSession
import os
import argparse
import mlflow
from mlflow.utils.environment import _mlflow_conda_env
from amazon_reviews.data_loader.data_loader import AmazonReviewsDataLoader
from amazon_reviews.recommender_model.amazon_product_recommender import AmazonProductRecommenderWrapper, \
    AmazonProductRecommender

PATH = '/dbfs/FileStore/shared_uploads/amazon_reviews'

def get_arguments():
    parser = argparse.ArgumentParser(description='reads default arguments')
    parser.add_argument('--run_id', metavar='run_id', type=str, help='Databricks run id')
    parser.add_argument('--job_id', metavar='job_id', type=str, help='Databricks job id')
    args = parser.parse_args()
    return args.run_id, args.job_id

git_sha = os.environ['GIT_SHA']
run_id, job_id = get_arguments()

DataLoader = AmazonReviewsDataLoader(path=PATH)
df = DataLoader.create_reviews_dataframe()

mlflow.set_experiment(experiment_name='/Shared/Amazon_recommender')
with mlflow.start_run(run_name="amazon-recommender") as run:
    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(df)
    recom_model = AmazonProductRecommender(spark_df=spark_df)
    recom_model.train()
    wrapped_model = AmazonProductRecommenderWrapper(recom_model.model)
    mlflow_run_id = run.info.run_id
    tags = {
        "GIT_SHA": git_sha,
        "MLFLOW_RUN_ID": mlflow_run_id,
        "DBR_JOB_ID": job_id,
        "DBR_RUN_ID": run_id,
    }
    mlflow.set_tags(tags)

    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["code/amazon_reviews-0.0.1-py3-none-any.whl",
                             "pyspark==3.3.0"
                             ],
        additional_conda_channels=None,
    )
    # this does not work because of pyspark:
    # https://docs.databricks.com/machine-learning/model-serving/private-libraries-model-serving.html

    mlflow.pyfunc.log_model("model",
                            python_model=wrapped_model,
                            conda_env=conda_env,
                            code_path=["/dbfs/upload/amazon-reviews/amazon_reviews-0.0.1-py3-none-any.whl"])

    model_version = mlflow.register_model(model_uri=f"runs:/{mlflow_run_id}/model",
                                          name='amazon-recommender',
                                          tags=tags)
