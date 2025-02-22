from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
import datetime
from wikipedia import summary

# Load Environment 
load_dotenv()

# Define Tools functions

def get_date_time(*args, **kwargs):
    now =  datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    try:
        return summary(query, sentences=2)
    except Exception as e:
        f"Couldn't find any information {str(e)}"

# Define tools 
tools = [
    Tool(
        name='Time',
        func=get_date_time,
        description='Useful when you need to find the current time.'
    ), 
    Tool(
        name='Wikipedia',
        func=search_wikipedia, 
        description='Useful when you need to find information about a topic.'
    )
]

# Create Prompt 
prompt = hub.pull("hwchase17/structured-chat-agent")

# Create llm 
llm = ChatGroq(model_name="Gemma2-9b-It")

# Create a structured Chat Agent with Conversation Buffer Memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create Chat Agent
agent = create_structured_chat_agent(
    llm=llm, 
    tools=tools, 
    prompt=prompt
)

# Create agent executor 
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools,
    verbose=True, 
    memory=memory, 
    handle_parsing_errors=True,
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
