import os
from pathlib import Path

import vertexai
from dotenv import load_dotenv

from simple_math_agent2.a2a_config import agent_card
from simple_math_agent2.executor import SimpleMathAgentExecutor

import logging

from google.genai import types
from vertexai.agent_engines.templates.a2a import A2aAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


ENGINE_ID = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "NULL_ENGINE_ID")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "NULL_PROJECT_ID")
REGION = os.environ.get("GOOGLE_CLOUD_LOCATION", "NULL_LOCATION")

DEPLOYMENT_BUCKET = os.environ.get("DEPLOYMENT_GCS", "NULL_DEPLOYMENT_BUCKET")
BUCKET_URI = f"gs://{DEPLOYMENT_BUCKET}"

logger.info("11111  Using Google Cloud Project ID: %s", PROJECT_ID)

# 1. Instantiate our custom wrapper
a2a_agent = SimpleMathAgentExecutor()
a2a_agent._init_agent()

# 2. Deploy it
# remote_agent = reasoning_engines.ReasoningEngine.create(
#     a2a_agent,
#     ...
# )


def main():
    vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
    client = vertexai.Client(
        project=PROJECT_ID,
        location=REGION,
        http_options=types.HttpOptions(api_version="v1beta1", base_url=f"https://{REGION}-aiplatform.googleapis.com")
    )

    print("Deploying Simple Math Agent to Agent Runtime...")
    print("This may take 3-5 minutes.")

    a2a_agent = A2aAgent(
        agent_card=agent_card,
        agent_executor_builder=lambda: SimpleMathAgentExecutor(),
    )

    remote_agent = client.agent_engines.create(
        agent=a2a_agent,
        config={
            "display_name": agent_card.name,
            "description": agent_card.description,
            "requirements": [
                "a2a-sdk>=1.0.0",
                "google-cloud-aiplatform[agent_engines,adk]>=1.156.0",
                "sse_starlette"
            ],
            "extra_packages": [
                "./simple_math_agent2",
            ],
            "http_options": {
                "base_url": f"https://us-central1-aiplatform.googleapis.com",
                "api_version": "v1beta1",
            },
            "staging_bucket": BUCKET_URI,
            "env_vars": {
                "GOOGLE_CLOUD_PROJECT_ID": PROJECT_ID,
                "GOOGLE_CLOUD_LOCATION": REGION,
                "MODEL_ARMOR_TEMPLATE": "tpl-test"
            }
        },
    )

    resource_name = remote_agent.api_resource.name
    print(f"\nDeployment complete!")
    print(f"Resource name: {resource_name}")

    env_path = Path(".env")
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    lines = [l for l in lines if not l.startswith("SIMPLE_MATH_AGENT_RESOURCE_NAME=")]
    lines.append(f"SIMPLE_MATH_AGENT_RESOURCE_NAME={resource_name}")
    env_path.write_text("\n".join(lines) + "\n")
    print("Written SIMPLE_MATH_AGENT_RESOURCE_NAME to .env")


if __name__ == "__main__":
    main()