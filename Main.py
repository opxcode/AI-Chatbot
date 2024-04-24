from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback

# Access the API key
load_dotenv()
try:
    api_key = os.getenv("api_key")
    if api_key == None:
        api_key = input("Enter your OpenAI API Key:\n")
except:
    # if there is no environment file
    api_key = input("Enter your OpenAI API Key:\n")



prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
output_parser = StrOutputParser()
llm = ChatOpenAI(api_key = api_key,model="gpt-3.5-turbo")

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    #Custom persona
    ("system", "You are a black belt brazillian jiu-jitsu coach."),
    ("user", """Answer the question based on the context.
    Context: "To pull guard, you need to step your left foot forward." To take down you need to grab collar"
    Question: {question}""")
    ])
chain = (
    prompt
    | llm
)

def initiate_chat():
    while True:
        question = input("How can i help? Type exit to quit.\n")
        if question.lower() == "exit":
            print("Bye!")
            break
        with get_openai_callback() as cb:
            result = chain.invoke({"question":{question}})
            #Check token used:
            print("--------------------------------------")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print("--------------------------------------")
            print(result.content)


initiate_chat()

#Prompt engineering + chat input


#Agent
#txt?
#RAG - answer question from document




