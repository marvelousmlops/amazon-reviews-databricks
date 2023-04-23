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
