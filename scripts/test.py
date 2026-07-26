import vertexai
from vertexai.preview import reasoning_engines

vertexai.init(project="project-6a0884c5-1380-4d43-808", location="us-central1")

# Load your deployed reasoning engine
remote_agent = reasoning_engines.ReasoningEngine(
    "projects/869928330868/locations/us-central1/reasoningEngines/1000750744590090240"
)

# Print the available methods and their expected inputs
for schema in remote_agent.operation_schemas():
    print(f"Exposed method: {schema['name']}")
    print(f"Expected input: {schema.get('parameters', 'No parameters expected')}\n")
# Call using ADK A2A payload or query format
# response = remote_agent.query(
#     message={"role": "user", "parts": [{"text": "Hi"}]}
# )
# print(response)

# 1. Define the A2A request payload
a2a_request = {
    "message": {
        "role": "user",
        "parts": [{"text": "What is 10 + 15?"}] 
    }
}

# 2. Define the context (an empty dict tells the ADK to generate a new task_id)
a2a_context = {}

# 3. Call the exposed A2A endpoint
print("Sending A2A task...")
response = remote_agent.on_message_send(
    request=a2a_request,
    context=a2a_context
)

print(response)