from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
 

load_dotenv()

#model = ChatOpenAI(model='gpt-3.5-turbo')
model = ChatGroq(model_name="Gemma2-9b-It")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a comedian who tells jokes about {topic}. Just give the jokes no extra text"),
        ("human", "Tell me {count} jokes"),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}")

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic": "ball", "count": 2})
print(result)

