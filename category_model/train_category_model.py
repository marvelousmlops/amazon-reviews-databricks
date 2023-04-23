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
    tags = {
        "GIT_SHA": git_sha,
        "MLFLOW_RUN_ID": mlflow_run_id,
        "DBR_JOB_ID": job_id,
        "DBR_RUN_ID": run_id,
    }

    model_version = mlflow.register_model(model_uri=f"runs:/{mlflow_run_id}/model",
                                          name='amazon-category-model',
                                          tags=tags)
