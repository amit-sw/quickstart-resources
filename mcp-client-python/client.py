import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

SYSTEM_PROMPT = (
    "You are an AI assistant that helps users by calling tools when necessary. "
    "Use the tools provided to you to answer user queries effectively. "
    "Before executing any SQL, remember that the schema is as follows:\n\n"
    """Based on the schema information provided, you have access to one table:

## **CompanyFounding** (public schema)

This table contains information about companies and has the following columns:

- **Symbol** (text, NOT NULL, Primary Key) - Company stock symbol
- **Security** (text, nullable) - Security name/description
- **GICS Sector** (text, nullable) - Global Industry Classification Standard sector
- **GICS Sub-Industry** (text, nullable) - GICS sub-industry classification
- **Headquarters Location** (text, nullable) - Company headquarters location
- **Date added** (text, nullable) - Date when the company was added
- **CIK** (bigint, nullable) - Central Index Key (SEC identifier)
- **Founded** (text, nullable) - Company founding date/year"""
    "When executing SQL, remember that table and column names are case-sensitive. "
    "If using Pinecone, do not use list-indexes tool; you know that the index name is 'books' and namespace is 'namespace1'. "
    "For Semantic Search, use the tool 'search-records' with the query parameter. "
    "If you don't know the answer, use the tools to find out."
)

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect(self, server_config: dict):
        """Connect to an MCP server from a configuration dictionary."""
        command = server_config.get("command")
        if not command:
            raise ValueError("Server configuration must include a 'command'")

        args = server_config.get("args", [])
        env = server_config.get("env", {})

        server_params = StdioServerParameters(command=command, args=args, env=env)
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-0",
            max_tokens=10000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        for content in response.content:
            if content.type == 'text':
                print(f"DEBUG: Received text content: {content.text}")
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                print(f"DEBUG: Received tool call request: {tool_name=} with args {tool_args=}")
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })
                print(f"DEBUG: Added tool call response: {result.content=}")

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-sonnet-4-0",
                    system=SYSTEM_PROMPT,
                    max_tokens=10000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file")
    parser.add_argument("--server", required=True, help="Name of the server to connect to")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config}")
        sys.exit(1)

    server_config = config.get("mcpServers", {}).get(args.server)
    if not server_config:
        print(f"Error: Server '{args.server}' not found in the configuration file.")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect(server_config)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
