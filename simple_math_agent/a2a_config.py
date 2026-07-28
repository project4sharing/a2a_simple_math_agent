import logging

# A2A
from a2a import types as a2a_types

# Agent Engine
from vertexai.agent_engines.templates.a2a import create_agent_card


logging.getLogger().setLevel(logging.INFO)


# Define a skill - a specific capability your agent offers
# Agents can have multiple skills for different tasks

simple_math_skill = a2a_types.AgentSkill(
    id="add_subtract_integers",
    name="Simple A2A Math Agent Skill",
    description="Perform addition and subtraction of two integers",
    # Tags for categorization and discovery
    # These help in agent marketplaces or registries
    tags=["math", "arithmetic"],
    # Examples show clients what kinds of requests work well
    # This is especially helpful for LLM-based clients
    examples=[
        "Add 1 and 2",
        "Subtract 3 from 1",
        "What is 5 plus 3?",
        "What is 10 minus 4?"
    ],
    input_modes=["text/plain"],
    output_modes=["text/plain"],
)

# Use the helper function to create a complete Agent Card
agent_card = create_agent_card(

    agent_name="Simple A2A Math Agent",
    description="An agent that can perform addition and subtraction of two integers.",
    skills=[simple_math_skill]
)