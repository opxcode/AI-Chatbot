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
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from datetime import date,timedelta
import streamlit as st
import customfunction
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage,FunctionMessage,HumanMessage,SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_function,format_tool_to_openai_function
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import functools
import json
from langgraph.graph import StateGraph, END


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
    speakingtone = st.text_input("set speaking style",value = "funny")
    sport = st.text_input("set sports specialisation",value = "brazillian jiu-jitsu")

#API key input overrides environment
api_key = openai_api_key
  
if not api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

llm = ChatOpenAI(api_key = api_key,model="gpt-3.5-turbo",temperature= 0.7)

#Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
embeddings_model = OpenAIEmbeddings(api_key = api_key,model="text-embedding-3-small")

#Context file supports only text files
doc_loader = DirectoryLoader('Context', glob="**/*.txt", loader_cls=TextLoader) 
docs = doc_loader.load()
documents = text_splitter.split_documents(docs)
#intialize vector store
vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings_model)
embeddings = {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}


#Context file supports only text files
technique_loader = TextLoader("Context/Techniques.txt")
technique_docs = technique_loader.load()
technique_documents = text_splitter.split_documents(technique_docs)
#intialize vector store
technique_vectorstore = DocArrayInMemorySearch.from_documents(technique_documents, embeddings_model)
technique_embeddings = {"context": technique_vectorstore.as_retriever(), "question": RunnablePassthrough()}

#Context file supports only text files
training_loader = TextLoader("Context/TrainingLog.txt")
training_docs = training_loader.load()
training_documents = text_splitter.split_documents(training_docs)
#intialize vector store
training_vectorstore = DocArrayInMemorySearch.from_documents(training_documents, embeddings_model)
training_embeddings = {"context": training_vectorstore.as_retriever(), "question": RunnablePassthrough()}

#Context file supports only text files
goals_loader = TextLoader("Context/Goals.txt")
goals_docs = goals_loader.load()
goals_documents = text_splitter.split_documents(goals_docs)
#intialize vector store
goals_vectorstore = DocArrayInMemorySearch.from_documents(goals_documents, embeddings_model)
goals_embeddings = {"context": goals_vectorstore.as_retriever(), "question": RunnablePassthrough()}

#Prompt Templates

prompt_context = ChatPromptTemplate.from_messages([
    ("user", """Answer the question based on the context. You should supplement it with your knowledge:
    Context: {context}
    Question: {question}""")
    ])

prompt_add = ChatPromptTemplate.from_messages([
    ("user", """From the given text, input the date in the format "DD MMM YYYY" followed by bullet points of what was done
    Text: {question}""")
    ])
prompt_persona = ChatPromptTemplate.from_messages([
    SystemMessage(content=f"""You are a {sport} expert. Respond in {speakingtone} style. 
                  If response involve steps, return it bullet format. When asked on what to do or how to do, suggest techniques.
                   Any questions regarding training or techniques , use the retriever tool.
                When analysing performance, identify the good or bad and suggest recommendation to improve.
                 """),
    MessagesPlaceholder(variable_name="messages"),
])

#Chain 
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

@tool
def retriever(query) -> str:
    """Retrieve data to answer training related questions"""
    response = chain_context.invoke(query)
    return response.content

@tool
def add_entry(query) -> str:
    """adds and write entry to training log"""
    result = chain_add.invoke(query)
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

tools = [retriever,add_entry]
functions = [convert_to_openai_function(t) for t in tools]
llm = llm.bind_functions(functions)
chain_persona = (
    prompt_persona
    |llm
)
tool_executor = ToolExecutor(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state['messages']
    print(messages)
    response = chain_persona.invoke(messages)
    # We return a list, because this will get added to the existing list
    print("___callmode___")
    print(response)
    print("_____")
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state):
    messages = state['messages']
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    print("___agent action___")
    print(action)
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    print("___tool result___")
    print(response)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent` where we start
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge('action', 'agent')

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

if user_prompt is not None and user_prompt != "":
    try:
        if api_key != None:
            with get_openai_callback() as cb:
                datequery = customfunction.calendar(user_prompt)
                if datequery is None:
                    query = user_prompt
                    inputs = {"messages": [HumanMessage(content=query)]}
                else:
                    query = user_prompt+" " + datequery
                    inputs = {"messages": [HumanMessage(content=query)]}
                # for output in app.stream(inputs):
                #     #stream() yields dictionaries with output keyed by node name
                #     for key, value in output.items():
                #         print(f"Output from node '{key}':")
                #         print("---")
                #         print(value)
                #     print("\n---\n")
                result = app.invoke(inputs)
                print("___Final result___")
                print(result)
             
                #Check token used:
                nl = "\n"
                token_usage = f"Total Tokens: {cb.total_tokens}{nl}Total Cost (USD): ${cb.total_cost:.8f}"
            st.session_state.chat_history.append({"user":user_prompt,"assistant":result['messages'][-1].content,"token_usage":token_usage})

        else:
            st.session_state.chat_history.append({"user":user_prompt,"assistant":"Please input API key or setup in environment","token_usage":""})
    except Exception as e:
        print(e)
        st.session_state.chat_history.append({"user":user_prompt,"assistant":"Error","token_usage":""})

for msg in st.session_state.chat_history:
   st.chat_message(name = "token_usage", avatar = "💎" ).write(msg["token_usage"])
   st.chat_message("user").write(msg["user"])
   st.chat_message("assistant").write(msg["assistant"])