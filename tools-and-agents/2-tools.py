from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool, Tool
from dotenv import load_dotenv

# Load the env 
load_dotenv()


# Tool functions 
def greet_user(name: str):
    return f'hello {name}!'

def reverse_string(text: str):
    return text[::-1]

def concatenate_string(text1: str, text2: str):
    return text1+text2

# Pydantic model for tool arguments
class ConcatenateStringsArgs(BaseModel):
    text1: str = Field(description="First String")
    text2: str = Field(description="Second String")

# Create Tools 
tools = [
    Tool(
        name="GreetUser",
        func=greet_user,
        description="Greet the user by name"
    ), 
    Tool(
        name="ReverseString",
        func=reverse_string,
        description="Reverse the given string"
    ), 
    # Use StructuredTool for more complex functions that require multiple input parameters.
    # StructuredTool allows us to define an input schema using Pydantic, ensuring proper validation and description.
    StructuredTool.from_function(
        name="ConcatenateStrings",
        func=concatenate_string,
        description="Concatenates two strings.",
        args_schema=ConcatenateStringsArgs
    ),
]

# Initialize the llm 
llm = ChatGroq(model_name="Gemma2-9b-It")

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the agent 
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create th executor 
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Test the agent
result = agent_executor.invoke({"input": "Greet Emma"})
print(f'Response for Greet Emma : {result}')

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)