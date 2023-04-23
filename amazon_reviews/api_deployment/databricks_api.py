import os
import requests

def databricks_host():
    dtb_url = os.environ["DATABRICKS_HOST"]
    dtb_url = dtb_url.strip("/")
    if dtb_url.startswith("https://"):
        dtb_url = dtb_url[8:]
    return dtb_url

def databricks_post_api(endpoint, json: dict):
    response = requests.post(
        f"https://{databricks_host()}/api/{endpoint}",
        headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
        json=json,
    )
    return response

def databricks_get_api(endpoint):
    response = requests.get(
        f"https://{databricks_host()}/api/{endpoint}",
        headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
    )
    return response

def databricks_put_api(endpoint, json):
    response = requests.put(
        f"https://{databricks_host()}/api/{endpoint}",
        headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
        json=json,
    )
    return response


def serve_ml_model_endpoint(endpoint_name, endpoint_config) -> requests.Response:
    """
    Args:
        endpoint_name (str): Name of the endpoint
        endpoint_config (dict): Endpoint config
    Returns:
        requests.Response

    Example of endpoint config:
        {"served_models": [{
                "model_name": "amazon-category-predictor",
                "model_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": False,
            }]
            }
    """
    response = databricks_get_api(f'2.0/serving-endpoints/{endpoint_name}')
    if response.status_code != 200 and 'RESOURCE_DOES_NOT_EXIST' in response.text:
        endpoint = '2.0/serving-endpoints'
        config = {
            "name": f"{endpoint_name}",
            "config": endpoint_config
        }
        ml_endpoint = databricks_post_api(endpoint, json=config)
    else:
        ml_endpoint = databricks_put_api(endpoint=f'2.0/serving-endpoints/{endpoint_name}/config',
                                         json=endpoint_config)
    return ml_endpoint