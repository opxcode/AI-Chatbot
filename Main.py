from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.tools.retriever import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from datetime import date,timedelta

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
llm = ChatOpenAI(api_key = api_key,model="gpt-3.5-turbo-1106",temperature= 0.7,top_p=1)

#Context file
loader = DirectoryLoader('./Context', glob="**/*.txt",loader_cls=TextLoader)
docs = loader.load()
#Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

#intialize vector store
embeddings_model = OpenAIEmbeddings(api_key = api_key,model="text-embedding-3-small")
vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings_model)

embeddings = {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}

#Prompt Templates
prompt_general = ChatPromptTemplate.from_messages([ 
    ("user", """ {question}""")
    ])
prompt_context = ChatPromptTemplate.from_messages([
    ("user", """Answer the question based on the context.
    Context: {context}
    Question: {question}""")
    ])

prompt_add = ChatPromptTemplate.from_messages([
    ("user", """From the given text, input the date in the format "DD MMM" followed by bullet points of what was done
    Text: {question}""")
    ])
#Chain
chain_general = (
    prompt_general
    | llm
)
chain_context = (
    embeddings
    |prompt_context
    | llm
)
chain_add = (
    prompt_add
    | llm
)
#Tools
Base_chat = Tool(
    name = 'General',
    func = chain_general.invoke,
    description= "all other questions"
)

#define date function AI does not have calendar
def calendar(question):
    now = date.today()
    if "today" in question.lower():
        now = now.strftime("%d %b %Y")
        datequery = f"today refer to {now}"
    elif "yesterday" in question.lower():
        yesterday = now - timedelta(days=1)
        now = yesterday.strftime("%d %b %Y")
        datequery = f"yesterday refer to {now}"
    elif "last week" in question.lower():
        last_monday = now - timedelta(days=now.weekday())
        last_sunday = last_monday + timedelta(days=6)
        start_date = last_monday.strftime("%d %b %Y")
        end_date = last_sunday.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    elif "this week" in question.lower():
        this_monday = now - timedelta(days=now.weekday())
        this_sunday = this_monday + timedelta(days=6)
        start_date = this_monday.strftime("%d %b %Y")
        end_date = this_sunday.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    elif "this month" in question.lower():
        this_month_start = date(now.year, now.month, 1)
        next_month = now.month + 1 if now.month < 12 else 1
        next_year = now.year if now.month < 12 else now.year + 1
        this_month_end = date(next_year, next_month, 1) - timedelta(days=1)
        start_date = this_month_start.strftime("%d %b %Y")
        end_date = this_month_end.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    elif "last month" in question.lower():
        last_month = now.month - 1 if now.month > 1 else 12
        last_year = now.year if now.month > 1 else now.year - 1
        last_month_start = date(last_year, last_month, 1)
        last_month_end = date(now.year, now.month, 1) - timedelta(days=1)
        start_date = last_month_start.strftime("%d %b %Y")
        end_date = last_month_end.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    elif "last year" in question.lower():
        last_year = now.year - 1
        last_year_start = date(last_year, 1, 1)
        last_year_end = date(last_year, 12, 31)
        start_date = last_year_start.strftime("%d %b %Y")
        end_date = last_year_end.strftime("%d %b %Y")
        datequery = f"date range is from {start_date} to {end_date}"
    else:
        return None
    return datequery

def training_questions(question):
    result = chain_context.invoke(question)
    return result

RAG_tool = Tool(
name = 'Retriver',
func = training_questions,
description= "to retrieve context to answer question on training")
#log workout function
def traininglog_create(question): 
    result = chain_add.invoke(question)
    """
    Writes the provided data to the specified text file.

    Args:
        filename (str): The name or path of the text file.
        data (str): The data to be written to the file.
    """
    data = result.content
    filename = "./Context/TrainingLog.txt"
    f = open(filename, "a")
    f.write("\n"+data)
    f.close()

AddEntry_tool = Tool(
    name = 'AddEntry',
    func = traininglog_create,
    description= "to add entry to training log"
)
tools = [Base_chat,RAG_tool,AddEntry_tool]

#Agent
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a very powerful and friendly assistant."),
        ("user", "{question}" ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True,max_iterations=3)

def initiate_chat():
    while True:
        question = input("How can i help? Type exit to end session.\n")
        datequery = calendar(question)
        query = f"{question} {datequery}"
        if question.lower() == "exit":
            print("Bye!")
            break
        with get_openai_callback() as cb:
            result = agent_executor.invoke({"question": query})
            #Check token used:
            print("--------------------------------------")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print("--------------------------------------")
            print(result["output"])
    
initiate_chat()





