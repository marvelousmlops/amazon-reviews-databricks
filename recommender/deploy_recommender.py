from amazon_reviews.api_deployment.databricks_api import serve_ml_model_endpoint
from mlflow.tracking.client import MlflowClient
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='reads default arguments')
    parser.add_argument('--run_id', metavar='run_id', type=str, help='Databricks run id')
    parser.add_argument('--job_id', metavar='job_id', type=str, help='Databricks job id')
    args = parser.parse_args()
    return args.run_id, args.job_id


run_id, job_id = get_arguments()

client = MlflowClient()
model_version = client.search_model_versions(f"name='amazon-recommender' and tag.DBR_RUN_ID = '{run_id}'")[0].version

client.transition_model_version_stage(name='amazon-recommender',
                                      version=model_version,
                                      stage='Production')

config = {
    "served_models": [{
        "model_name": "amazon-recommender",
        "model_version": f"{model_version}",
        "workload_size": "Small",
        "scale_to_zero_enabled": False,
    }]
    }

serve_ml_model_endpoint(endpoint_name='amazon-recommender', endpoint_config=config)
