from amazon_reviews.api_deployment.databricks_api import databricks_get_api
import os
import requests

response = requests.post(
            f"https://{os.environ['DATABRICKS_HOST']}/serving-endpoints/amazon-category-model/invocations",
            headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
            json={"inputs": ["Oh man, the best cereals I've ever had! A lot of nuts, and I love nuts.", 
                             "Best face serum, contains vitamin C, pantenol, and your skin glows!"]})

print("Response status:", response.status_code)
print("Reponse text:", response.text)

model_input = {
    'customer_id': 'abcdefg12345678abcdefg',
    'basket': ['B00000J0FW'], # Sassy Who Loves Baby Photo Book; baby products	
}

response = requests.post(
            f"https://{os.environ['DATABRICKS_HOST']}/serving-endpoints/amazon-recommender/invocations",
            headers={"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"},
            json={"inputs": model_input})
            
print("Response status:", response.status_code)
print("Reponse text:", response.text)

