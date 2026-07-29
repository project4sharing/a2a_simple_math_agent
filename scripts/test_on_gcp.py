import vertexai
from vertexai.preview import reasoning_engines
import os
import asyncio
from pprint import pprint
from pathlib import Path
from dotenv import load_dotenv
import logging

from google.genai import types
from a2a.types import SendMessageRequest, Message, Part

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
ENGINE_ID = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "NULL_ENGINE_ID")
AGENT_NAME = os.environ.get("SIMPLE_MATH_AGENT_RESOURCE_NAME", "NULL_SIMPLE_MATH_AGENT_RESOURCE_NAME")

async def main():

    logger.info(f"#####  {env_path}")
    logger.info("#####  Using Google Cloud Project ID: %s", PROJECT_ID)
    logger.info("#####  Using Google Cloud Location: %s", LOCATION)
    logger.info("#####  Using Model Armor Template: %s", MODEL_ARMOR_TEMPLATE)
    logger.info("#####  Full Model Armor Template: %s", FULL_MODEL_ARMOR_TEMPLATE_NAME)
    logger.info("#####  Agent Name: %s", AGENT_NAME)

    client = vertexai.Client(
        project=PROJECT_ID,
        location=LOCATION,
        http_options=types.HttpOptions(api_version="v1beta1"),
    )
    remote_agent = client.agent_engines.get(name=AGENT_NAME)

# sanity check — should list on_message_send, on_get_task, etc.
    print(remote_agent.operation_schemas())

    request = SendMessageRequest(
        message=Message(
            message_id="msg-1",
            role="ROLE_USER",
            parts=[Part(text="What is 1 + 23?")],
        ),
        # configuration=MessageSendConfiguration(return_immediately=False),
    )

    response = await remote_agent.on_message_send(request=request)
    print(response)

asyncio.run(main())
