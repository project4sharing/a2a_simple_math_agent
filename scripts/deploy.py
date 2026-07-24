import os
from pathlib import Path

import vertexai
from dotenv import load_dotenv
from google.genai import types
from vertexai.preview.reasoning_engines import A2aAgent

from simple_math_agent.a2a_config import agent_card
from simple_math_agent.executor import SimpleMathAgentExecutor

load_dotenv()

ENGINE_ID = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "NULL_ENGINE_ID")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "NULL_PROJECT_ID")
REGION = os.environ.get("GOOGLE_CLOUD_LOCATION", "NULL_LOCATION")

DEPLOYMENT_BUCKET = os.environ.get("DEPLOYMENT_GCS", "NULL_DEPLOYMENT_BUCKET")
BUCKET_URI = f"gs://{DEPLOYMENT_BUCKET}"

a2a_agent = A2aAgent(
    agent_card=agent_card,
    agent_executor_builder=SimpleMathAgentExecutor,
)


def main():
    vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
    client = vertexai.Client(
        project=PROJECT_ID,
        location=REGION,
        http_options=types.HttpOptions(api_version="v1beta1"),
    )

    print("Deploying Simple Math Agent to Agent Runtime...")
    print("This may take 3-5 minutes.")

    remote_agent = client.agent_engines.create(
        agent=a2a_agent,
        config={
            "display_name": agent_card.name,
            "description": agent_card.description,
            "requirements": [
                "google-cloud-aiplatform[agent_engines,adk]>=1.91.0",
                "a2a-sdk==0.3.26",
                "google-adk>=2.4.0",
                "cloudpickle",
                "pydantic"
            ],
            "extra_packages": [
                "./simple_math_agent",
            ],
            "http_options": {
                "api_version": "v1beta1",
            },
            "staging_bucket": BUCKET_URI,
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