import os

import autogen
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configuration setup for AutoGen using environment variables
config_list = [
    {
        "model": "gemini-2.5-flash",
        "api_key": os.environ["GEMINI_API_KEY"],
        "api_type": "google",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
}

# Define the Assistant Agent (AI Reasoning Engine)
assistant = autogen.AssistantAgent(
    name="technical_assistant",
    llm_config=llm_config,
    system_message="You are a concise software architect specializing in distributed systems.",
)

# Define the User Proxy Agent (Executes code & acts as human/system proxy)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Fully automated loop
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config={"use_docker": False},  # Runs locally
)

if __name__ == "__main__":
    # Initiate the in-process two-agent conversation
    user_proxy.initiate_chat(
        assistant,
        message="Explain why A2A HTTP messaging is useful for microservices, in 2 sentences. End your answer with TERMINATE.",
    )
