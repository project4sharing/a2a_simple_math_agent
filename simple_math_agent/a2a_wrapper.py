# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai
from google.genai import types

from pathlib import Path
from dotenv import load_dotenv

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, verbose=True)

engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "NULL_ENGINE_ID")
logger.info("#####  Using Google Cloud Agent Engine ID: %s", engine_id)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "NULL_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "NULL_LOCATION")
MODEL_ARMOR_TEMPLATE = os.environ.get("MODEL_ARMOR_TEMPLATE", "NULL_MODEL_ARMOR_TEMPLATE")
BUCKET_URI = os.environ.get("DEPLOYMENT_GCS", "NULL_DEPLOYMENT_GCS")
FULL_MODEL_ARMOR_TEMPLATE_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{MODEL_ARMOR_TEMPLATE}"

logger.info(f"#####  {env_path}")
logger.info("#####  Using Google Cloud Project ID: %s", PROJECT_ID)
logger.info("#####  Using Google Cloud Location: %s", LOCATION)
logger.info("#####  Using Model Armor Template: %s", MODEL_ARMOR_TEMPLATE)
logger.info("#####  Full Model Armor Template: %s", FULL_MODEL_ARMOR_TEMPLATE_NAME)

ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com"


# !gsutil mb -l $LOCATION -p $PROJECT_ID $BUCKET_URI

# Initialize Agent Platform session
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET_URI,
    api_endpoint=ENDPOINT,  # This directs requests to the {$ENV} endpoint
)

# Initialize the Gen AI client using http_options
# The parameter customizes how the Agent Platform client communicates with Google Cloud's backend services.
# It's used here to access new, pre-release features.
client = vertexai.Client(
    project=PROJECT_ID,
    location=LOCATION,
    http_options=types.HttpOptions(api_version="v1beta1", base_url=f"{ENDPOINT}/"),
)