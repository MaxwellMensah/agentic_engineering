import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Define the path to your server and virtual environment python interpreter
SERVER_PARAMS = StdioServerParameters(
    command="/home/maxie/Documents/agentic_engineering/.pixi/envs/default/bin/python",
    args=["/home/maxie/Documents/agentic_engineering/mcp/fraud_mcpserver.py"],
)


async def run_fraud_agent(document_text: str):
    """Real-time fraud agent calling the local MCP server over STDIO."""
    print("🤖 Agent initiating real-time fraud analysis via MCP...")

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection with MCP server
            await session.initialize()

            # Call the 'analyze' tool exposed by fraud_mcpserver.py
            response = await session.call_tool(
                "analyze", arguments={"document_text": document_text}
            )

            # Display response
            print("\n✅ MCP Tool Response Received:")
            print(response.content[0].text)


if __name__ == "__main__":
    sample_document = (
        "Invoice #99214 for monthly cloud infrastructure ($4,250.00). "
        "Approved by J. Smith (IT Director)."
    )
    asyncio.run(run_fraud_agent(sample_document))
