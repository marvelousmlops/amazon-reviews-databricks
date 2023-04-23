import requests
from amazon_reviews.api_deployment.databricks_api import databricks_host


def retrieve_databricks_token(client_id, client_secret, tenant, lifetime=15768000) -> requests.Response:
    """Retrieve token using MS login API

    Args:
        client_id (str): SPN client id
        client_secret (str): SPN client secret
        tenant (str): tenant
        lifetime (str): token lifetime in sec, default 15768000, which is 1/2 year

    Returns:
        requests.Response
    """
    body = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default",
        "grant_type": "client_credentials",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    get_token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    token = requests.post(get_token_url, data=body, headers=headers).json()

    token = requests.post(
            f"https://{databricks_host()}/api/2.0/token/create",
            headers={"Authorization": f"Bearer {token['access_token']}"},
            json={"comment": "no comment",
                  "lifetime_seconds": f'{lifetime}'}
        )
    return token.json()['token_value']