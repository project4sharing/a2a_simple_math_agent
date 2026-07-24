from google.adk.agents import LlmAgent
# from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.planners import BuiltInPlanner

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "NULL_ENGINE_ID")
logger.info("#####  Using Google Cloud Agent Engine ID: %s", engine_id)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "NULL_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "NULL_LOCATION")
MODEL_ARMOR_TEMPLATE = os.environ.get("MODEL_ARMOR_TEMPLATE", "NULL_MODEL_ARMOR_TEMPLATE")

FULL_MODEL_ARMOR_TEMPLATE_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{MODEL_ARMOR_TEMPLATE}"

logger.info("#####  Using Google Cloud Project ID: %s", PROJECT_ID)
logger.info("#####  Using Google Cloud Location: %s", LOCATION)
logger.info("#####  Using Model Armor Template: %s", MODEL_ARMOR_TEMPLATE)
logger.info("#####  Full Model Armor Template: %s", FULL_MODEL_ARMOR_TEMPLATE_NAME)

def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of the two integers.
    """
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The difference of the two integers.
    """
    return a - b

# To use local OpenAI-compatible endpoint
# llm = LiteLlm(
#     model="LOCAL",              # "openai/" prefix = OpenAI-compatible endpoint
#     api_base="http://localhost:8080/v1",  # note: api_base, not base_url
#     api_key="NONE",
# )

#Set default model
LlmAgent.set_default_model("gemini-2.5-flash")



# To use Gemini 2.5 Flash model instead of local OpenAI-compatible endpoint
# llm = LiteLlm(model="gemini-2.5-flash") 
llm = "gemini-2.5-flash" 
# Change env.sample to .env so it will determine whether to use Gemini (AI Studio / Gemini API)or enterprise Vertex with google project

root_agent = LlmAgent(
    name="simple_math_agent",
    model=llm,
    description="Agent to answer questions about addition and subtraction of two integers.",
    instruction="""You are a helpful agent who can answer user questions about the sum and difference of two integers.
    When a user asks for the sum or difference of two integers:
    1. Validate that the input is in the form of two integers.
    2. If the input is valid, call the appropriate tool (add or subtract) to compute the result.
    3. Return the result to the user.
    Example Query: "Add 1 and 2"
    Example Response: "The sum of 1 and 2 is 3."
    Example Query: "Subtract 3 from 1"
    Example Response: "The difference of 3 and 1 is 2."
    """,
    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=256,
        # safety_settings=types.SafetySetting(
        #     category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        #     threshold=types.HarmBlockThreshold.BLOCK_NONE
        # ),
        # Add Model Armor Config
        model_armor_config=types.ModelArmorConfig(
            prompt_template_name=FULL_MODEL_ARMOR_TEMPLATE_NAME,
            response_template_name=FULL_MODEL_ARMOR_TEMPLATE_NAME
        ),
    ),
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024
        )
    ),
    include_contents = "default",
    tools=[add, subtract]
)
