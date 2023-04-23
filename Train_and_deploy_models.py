# Databricks notebook source
# MAGIC %pip install /dbfs/upload/amazon-reviews/amazon_reviews-0.0.1-py3-none-any.whl

# COMMAND ----------

# Train category model

# COMMAND ----------

from amazon_reviews.data_loader.data_loader import AmazonReviewsDataLoader
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlflow
import argparse
import os

PATH = '/dbfs/FileStore/shared_uploads/amazon_reviews'

# COMMAND ----------

DataLoader = AmazonReviewsDataLoader(path=PATH)
df = DataLoader.create_reviews_dataframe()

# COMMAND ----------

df.head()

# COMMAND ----------

nltk.download('stopwords')

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer()),
    ('lr', LogisticRegression(max_iter=3000))
])

mlflow.sklearn.autolog()

mlflow.set_experiment(experiment_name='/Shared/Amazon_category_model')
with mlflow.start_run(run_name='amazon-category-model') as run:

    pipeline.fit(df.Text.values, df.Cat1.values)
    mlflow_run_id = mlflow.active_run().info.run_id

    model_version = mlflow.register_model(model_uri=f"runs:/{mlflow_run_id}/model",
                                          name='amazon-category-model')

# COMMAND ----------

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

from amazon_reviews.api_deployment.databricks_api import serve_ml_model_endpoint


endpoint_name='amazon-category-model'

config = {
    "served_models": [{
        "model_name": "amazon-category-model",
        "model_version": f"{dict(model_version)['version']}",
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }]
    }

serve_ml_model_endpoint(endpoint_name=endpoint_name, endpoint_config=config)

# COMMAND ----------

from amazon_reviews.api_deployment.databricks_api import databricks_get_api
databricks_get_api(f'2.0/serving-endpoints/{endpoint_name}').text

# COMMAND ----------

import requests

endpoint = f"https://{os.environ['DATABRICKS_HOST']}/serving-endpoints/{endpoint_name}/invocations"
response = requests.post(
            f"{endpoint}",
            headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
            json={"inputs": ["Oh man, the best cereals I've ever had! A lot of nuts, and I love nuts.", "Best face serum, contains vitamin C, pantenol, and your skin glows!"]})

print("Response status:", response.status_code)
print("Reponse text:", response.text)

# COMMAND ----------

# Model based collaborative filtering; 
# https://medium.com/@gazzaazhari/model-based-collaborative-filtering-systems-with-machine-learning-algorithm-d5994ae0f53b

# COMMAND ----------

from pyspark.sql import functions as F
from sklearn.decomposition import TruncatedSVD
import numpy as np
import mlflow


class AmazonProductRecommenderWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        """Initializes the model in wrapper.
        Args:
            model: an AmazonProductRecommender object.
        Returns:
            None
        """
        self.model = model

    def predict(self, context, model_input: dict):
        """Returns the model's prediction based on the model input.
        Args:
            model_input: dictionary in format: {'basket': List[str], 'customer_id': str}.
        Returns:
            The prediction of the model based on the model input.
        """
        model_input = {'customer_id': str(model_input['customer_id']),
                       'basket': list(model_input['basket'])}
        return self.model[model_input['basket'][0]]


class AmazonProductRecommender:
    """
    Amazon product recommender: recommends list of products based on product id
    Model based collaborative filtering; inspired by:
    https://medium.com/@gazzaazhari/model-based-collaborative-filtering-systems-with-machine-learning-algorithm-d5994ae0f53b

    """
    def __init__(self, spark_df):
        """
        Creates dictionary of form {'product1' : ['product7', 'product3'],
                                    'product2': ['product5', 'product6']}
        :param spark_df: spark dataframe
        """
        self.spark_df = spark_df
        self.model = {}

    def train(self):
        active_user_counts = self.spark_df.groupBy("userId").agg(F.count('*').alias("user_count")).filter(
            "user_count >= 5")
        self.spark_df = self.spark_df.join(active_user_counts, "userId", "inner")
        product_counts = self.spark_df.groupBy("productId").agg(F.count('*').alias("product_count")).filter(
            "product_count >= 30")
        self.spark_df = self.spark_df.join(product_counts, "productId", "inner").drop("user_count", "product_count")
        spark_df_new = self.spark_df.select('userId', 'productId', 'Score').withColumn(
            "Score", self.spark_df.Score.cast("float"))
        spark_df_pivoted = spark_df_new.groupBy("userId").pivot("productId").avg("Score").na.fill(0)
        ratings_matrix = spark_df_pivoted.toPandas()
        ratings_matrix = ratings_matrix.set_index('userId')
        X = ratings_matrix.T
        SVD = TruncatedSVD(n_components=10)
        decomposed_matrix = SVD.fit_transform(X)
        correlation_matrix = np.corrcoef(decomposed_matrix)
        product_names = list(X.index)
        for product in product_names:
            product_id = product_names.index(product)
            recom = list(np.array(X.index)[(-correlation_matrix[product_id]).argsort()[1:10]])
            self.model[product] = recom
        return self

# COMMAND ----------

from pyspark.sql import SparkSession
from amazon_reviews.recommender_model.amazon_product_recommender import AmazonProductRecommenderWrapper, \
    AmazonProductRecommender
from mlflow.utils.environment import _mlflow_conda_env

mlflow.set_experiment(experiment_name='/Shared/Amazon_recommender')
with mlflow.start_run(run_name="amazon-recommender") as run:
    spark = SparkSession.builder.getOrCreate()
    spark_df=spark.createDataFrame(df)
    recom_model = AmazonProductRecommender(spark_df=spark_df)
    recom_model.train()
    wrapped_model = AmazonProductRecommenderWrapper(recom_model.model)
    mlflow_run_id = run.info.run_id

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
                            conda_env = conda_env,
                            code_path = ["/dbfs/upload/amazon-reviews/amazon_reviews-0.0.1-py3-none-any.whl"])

    model_version = mlflow.register_model(model_uri=f"runs:/{mlflow_run_id}/model",
                                          name='amazon-recommender')

# COMMAND ----------

from amazon_reviews.api_deployment.databricks_api import serve_ml_model_endpoint
endpoint_name='amazon-recommender'

config = {
    "served_models": [{
        "model_name": "amazon-recommender",
        "model_version": f"{dict(model_version)['version']}",
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }]
    }

serve_ml_model_endpoint(endpoint_name=endpoint_name, endpoint_config=config)

# COMMAND ----------

from amazon_reviews.api_deployment.databricks_api import databricks_get_api
databricks_get_api(f'2.0/serving-endpoints/{endpoint_name}').text

# COMMAND ----------

import requests
model_input = {
    'customer_id': 'abcdefg12345678abcdefg',
    'basket': ['B00000J0FW'], # Sassy Who Loves Baby Photo Book; baby products	
}

url = f"https://{os.environ['DATABRICKS_HOST']}/serving-endpoints/{endpoint_name}/invocations"


headers = {'Authorization': f"Bearer {os.environ['DATABRICKS_TOKEN']}", 'Content-Type': 'application/json'}
data_json = {'inputs': model_input}
response = requests.request(method='POST', headers=headers, url=url, json=data_json)
if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

print(response.text)
# {"predictions": ["B00005MKYI", "B000056HMY", "B000056JHM", "B000046S2U", "B000056JEG", "B000099Z9K", "B00032G1S0", "B00005V6C8", "B0000936M4"]}
# B00005MKYI: Deluxe Music In Motion Developmental Mobile, B000056HMY: The First Years Nature Sensations Lullaby Player; B000056JHM:Bebe Sounds prenatal gift set

# COMMAND ----------

df[df.productId=='B00000J0FW']

# COMMAND ----------

df[df.productId=='B00005MKYI']

# COMMAND ----------

df[df.productId=='B000056JHM']

# COMMAND ----------

recom_model.model
