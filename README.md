# Getting started
## 1. Notebook
If you want to run a provided notebook Train_and_deploy_models.py in Databricks, you need to do the following first:
```
export DATABRICKS_HOST=<DATABRICKS_HOST> 
export DATABRICKS_TOKEN=<DATABRICKS_TOKEN>
git clone https://github.com/marvelousmlops/amazon-reviews-databricks.git
cd amazon-reviews-databricks
python3 setup.py bdist_wheel
dbfs rm dbfs:/upload/amazon_reviews/amazon-reviews-0.0.1-py3-none-any.whl
dbfs cp dist/amazon_reviews-0.0.1-py3-none-any.whl dbfs:/upload/amazon_reviews/amazon-reviews-0.0.1-py3-none-any.whl
```

IMPORTANT: DATABRICKS_HOST value should have format: https://< your workspace >/.
It must have https:// and also / in the end.

## 2. Using Github Actions Workflow (and Databricks on Azure)

### Step 1: get SPN and Databricks token for it
First of all, you need to have a service principal that has admin permissions on Databricks and generate a token for it.
This documentation might be useful: https://docs.databricks.com/dev-tools/service-principals.html#language-curl

Note: /api/2.0/token-management/on-behalf-of/tokens is not enabled for Azure Databricks.
A workaround (given that you have SPN with Databricks admin permissions and its client id, client secret and tenant):
```
from amazon_reviews.utils.databricks_token import retrieve_databricks_token
import os
os.environ['DATABRICKS_HOST'] = <DATABRICKS_HOST>
token = retrieve_databricks_token(client_id='', client_secret='', tenant='')
```
DATABRICKS_HOST value should have format: https://< your workspace >/.

### Step 2: create keyvault backed secret scope and add databricks token to keyvault
DatabricksToken is referred in line 66 of both category_model.json.j2 and recommender.json.j2 as {{secrets/keyvault/DatabricksToken}}

Create keyvault backed secret scope called "keyvault" or, if such scope already exists 
and has a different name, update line 66 in both category_model.json.j2 and recommender.json.j2.

This documentation explains how to create a secret scope: https://learn.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes.

Add DatabricksToken to keyvault as explained here: https://learn.microsoft.com/en-us/azure/key-vault/secrets/quick-create-cli.

### Step 3: Repository setup
First of all, fork https://github.com/marvelousmlops/amazon-reviews-databricks
In the forked repository that you now own, create repository secrets DATABRICKS_HOST and DATABRICKS_TOKEN that you have created in step 1.

### Step 4: trigger workflow run
You are now ready to trigger the workflow
