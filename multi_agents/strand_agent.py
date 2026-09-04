import os

from dotenv import load_dotenv
from strands import Agent
from strands.models import GeminiModel

load_dotenv()  # Load environment variables from .env file


# Initialize Gemini natively via Strands
model = GeminiModel(
    model_id="gemini-2.5-flash",
    client_args={"api_key": os.environ["GEMINI_API_KEY"]},
)

# Define specialized Researcher agent
researcher = Agent(
    name="researcher",
    system_prompt="You are an expert technical researcher. Gather concise, factual insights on the given topic.",
    model=model,
)

# Define Writer agent with Researcher passed directly as a tool
writer = Agent(
    name="writer",
    tools=[researcher],  # Agent-as-Tool pattern
    system_prompt=(
        "You are a technical writer. First, delegate research to the researcher agent. "
        "Then synthesize their output into a clear, 2-paragraph summary."
    ),
    model=model,
)

# Run the orchestration
response = writer("Explain the impact of Quantum Computing on modern cryptography.")
print(response)
