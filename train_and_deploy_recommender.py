from pyspark.sql import functions as F
import numpy as np
import mlflow
from pyspark.sql import SparkSession
from amazon_reviews.data_loader.data_loader import AmazonReviewsDataLoader
from amazon_reviews.recommender_model.amazon_product_recommender import AmazonProductRecommenderWrapper
from amazon_reviews.api_deployment.databricks_api import serve_ml_model_endpoint
from mlflow.utils.environment import _mlflow_conda_env

PATH = '/dbfs/FileStore/shared_uploads/amazon_reviews'

DataLoader = AmazonReviewsDataLoader(path=PATH)
df = DataLoader.create_reviews_dataframe()

mlflow.set_experiment(experiment_name='/Shared/Amazon_recommender')
with mlflow.start_run(run_name="amazon-recommender") as run:
    spark = SparkSession.builder.getOrCreate()
    spark_df=spark.createDataFrame(df)
    recom_model = AmazonProductRecommenderWrapper(spark_df=spark_df)
    recom_model.train()
    mlflow_run_id = run.info.run_id

    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["code/amazon_reviews-0.0.1-py3-none-any.whl",
                             "pyspark==3.3.0"
                             ],
        additional_conda_channels=None,
    )

    mlflow.pyfunc.log_model("model",
                            python_model=wrapped_model,
                            conda_env = conda_env,
                            code_path = ["/dbfs/upload/amazon-reviews/amazon_reviews-0.0.1-py3-none-any.whl"])

    model_version = mlflow.register_model(model_uri=f"runs:/{mlflow_run_id}/model",
                                          name='amazon-recommender')

config = {
    "served_models": [{
        "model_name": "amazon-recommender",
        "model_version": f"{dict(model_version)['version']}",
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }]
    }

serve_ml_model_endpoint(endpoint_name='amazon-recommender', endpoint_config=config)