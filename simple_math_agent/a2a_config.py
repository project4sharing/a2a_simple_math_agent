from a2a.types import AgentSkill
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

simple_math_skill = AgentSkill(
    id="add_subtract_integers",
    name="Simple Math",
    description="Perform addition and subtraction of two integers",
    tags=["math", "arithmetic"],
    examples=[
        "Add 1 and 2",
        "Subtract 3 from 1",
        "What is 5 plus 3?",
        "What is 10 minus 4?",
    ],
    input_modes=["text/plain"],
    output_modes=["text/plain"],
)

agent_card = create_agent_card(
    agent_name="Simple Math Agent",
    description="An agent that can perform addition and subtraction of two integers.",
    skills=[simple_math_skill],
)