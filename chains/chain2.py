from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
 

load_dotenv()

#model = ChatOpenAI(model='gpt-3.5-turbo')
model = ChatGroq(model_name="Gemma2-9b-It")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {count} jokes"),
    ]
)

new_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x)) #**x unpacking dictionary
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

runnable_chain_seq = RunnableSequence(
    first=new_prompt, 
    middle=[invoke_model], 
    last=parse_output
)

# Run the chain
response = runnable_chain_seq.invoke({"topic": "lawyers", "count": 2})

# Output
print(response)