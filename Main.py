from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.globals import set_verbose
# Access the API key
load_dotenv()
try:
    api_key = os.getenv("api_key")
    if api_key == None:
        api_key = input("Enter your OpenAI API Key:\n")
except:
    # if there is no environment file
    api_key = input("Enter your OpenAI API Key:\n")

#output_parser = StrOutputParser()
llm = ChatOpenAI(api_key = api_key,model="gpt-3.5-turbo",temperature= 0.7)

#Context file
loader = DirectoryLoader('Context', glob="**/*.txt",loader_cls=TextLoader)
docs = loader.load()
#Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

#intialize vector store
embeddings_model = OpenAIEmbeddings(api_key = api_key,model="text-embedding-3-small")
vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings_model)

embeddings = {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}

#Custom persona
persona = "You are a friendly and energetic black belt brazillian jiu-jitsu coach."
#Prompt Templates
prompt_knowledegebase = ChatPromptTemplate.from_messages([
    ("system",persona),
    ("user", """Answer the question based on the context.
    Context: {context}
    Question: {question}""")
    ])
prompt_add = ChatPromptTemplate.from_messages([
    ("user", """From the given text, iutput the  date in the format "DD MMM" followed by bullet points of what was done
    Text: {question}""")
    ])

#Chains
chain = (
    embeddings
    |prompt_knowledegebase
    | llm
)
chain_add = (
    embeddings
    |prompt_add
    | llm
)

#log workout function
def traininglog_create(data):
    """
    Writes the provided data to the specified text file.

    Args:
        filename (str): The name or path of the text file.
        data (str): The data to be written to the file.
    """
    filename = "Context/TrainingLog.txt"
    f = open(filename, "a")
    f.write("\n"+data)
    f.close()

#TODO: need to add time function, AI model does not calculate corectly
#TODO: Persist memory

def initiate_chat():
   
    while True:
        question = input("How can i help? Type exit to quit.\n")
        if question.lower() == "exit":
            print("Bye!")
            break
        with get_openai_callback() as cb:
            if "new entry" in question:
                result = chain_add.invoke(question)
                traininglog_create(result.content)
            else:    
                result = chain.invoke(question)
            #Check token used:
            print("--------------------------------------")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print("--------------------------------------")
            print(result.content)

initiate_chat()





