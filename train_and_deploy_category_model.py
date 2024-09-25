from amazon_reviews.data_loader.data_loader import AmazonReviewsDataLoader
from amazon_reviews.api_deployment.databricks_api import serve_ml_model_endpoint
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlflow

PATH = '/dbfs/FileStore/shared_uploads/amazon_reviews'

DataLoader = AmazonReviewsDataLoader(path=PATH)
df = DataLoader.create_reviews_dataframe()

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

config = {
    "served_models": [{
        "model_name": "amazon-category-model",
        "model_version": f"{dict(model_version)['version']}",
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }]
    }

serve_ml_model_endpoint(endpoint_name='amazon-category-model', endpoint_config=config)