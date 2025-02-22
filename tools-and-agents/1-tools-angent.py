from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
import datetime

# Load environment variables
load_dotenv()

# Function to get current time for a given country or city
def get_current_time(location: str):
    try:
        now = datetime.datetime.now()  
        return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format

    except Exception as e:
        return f"Error fetching time : {str(e)}"

# List of tools available to the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Use when you need to know the current time.",
    ),
]

# Pull the prompt from LangChain's hub
prompt = hub.pull("hwchase17/react")

# Initialize the LLM model
llm = ChatGroq(model_name="Gemma2-9b-It")

# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,  # The agent decides which tool to call
    prompt=prompt,
    stop_sequence=True,
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,  # Executor calls the selected tool
    verbose=True,
)

# Test: Ask for the time in Berlin
result = agent_executor.invoke({"input": "What is the time now?"})
print(f"\n Result: {result['output']}")
