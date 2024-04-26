from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
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
import streamlit as st

#Streamlit
st.set_page_config(page_title="Personal Assistant", page_icon= "🤖")
st.markdown("As your personal coach, I can answer questions on your training needs")
example1 = ''':blue[To add entry into logfile, add entry keyword required]  
[Example]  
add entry, On 24 April, I did....  
add entry: Today's training felt good....  
'''
example2 = ''':blue[To ask question from training logfile]  
[Example]  
What did I do on 24 Apr  
How was training last week?  
'''
st.caption(example1)
st.caption(example2)
st.divider()
# Access the API key
load_dotenv()
try:
    api_key = os.getenv("api_key")
except:
    # if there is no environment file
    api_key = None

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
user_prompt = st.chat_input("Type Here")

with st.sidebar:
    try:
        st.image("Img/BrandName.png")
    except:
       st.subheader("ChatBot")
    openai_api_key = st.text_input("OpenAI API Key",value=api_key, key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    directory = st.text_input("Directory for context files",value = "Context",help = "default directory is Context\n Type to replace")
    logfile = st.text_input("File for update",value = "Context/TrainingLog.txt",help = "default directory is Context/TrainingLog.txt\n Type to replace")
    "[Readme](https://github.com/opxcode/AI-Chatbot)"
    speakingtone = st.text_input("set speaking style",value = "hip-hop")
    sport = st.text_input("set sports specialisation",value = "brazillian jiu-jitsu")

#API key input overrides environment
api_key = openai_api_key
  
if not api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(api_key = api_key,model="gpt-3.5-turbo-1106",temperature= 0.7)

#Context file supports only text files
loader = DirectoryLoader(directory, glob="**/*.txt",loader_cls=TextLoader)
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
    filename = logfile
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
    [ #set custom persona
        ("system","You are a very powerful assistant and a {sport} expert. You speak in {speakingtone} style. If the user ask who are you, say you a renowned {sport} coach who trains atheletes."),
        ("user", "{question}" ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True, max_iterations=3)

if user_prompt is not None and user_prompt != "":
    try:
        if api_key != None:
            with get_openai_callback() as cb:
                datequery = calendar(user_prompt)
                if datequery is None:
                    query = user_prompt
                else:
                    query = user_prompt+" " + datequery
                result = agent_executor.invoke({"question": query,"speakingtone":speakingtone,"sport":sport})
                #Check token used:
                nl = "\n"
                token_usage = f"Total Tokens: {cb.total_tokens}{nl}Total Cost (USD): ${cb.total_cost:.8f}"
            st.session_state.chat_history.append({"user":user_prompt,"assistant":result["output"],"token_usage":token_usage})
        else:
            st.session_state.chat_history.append({"user":user_prompt,"assistant":"Please input API key or setup in environment","token_usage":""})
    except Exception as e:
        print(e)
        st.session_state.chat_history.append({"user":user_prompt,"assistant":"Error","token_usage":""})

for msg in st.session_state.chat_history:
   st.chat_message(name = "token_usage", avatar = "💎" ).write(msg["token_usage"])
   st.chat_message("user").write(msg["user"])
   st.chat_message("assistant").write(msg["assistant"])




