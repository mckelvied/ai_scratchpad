import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import Literal
from dotenv import load_dotenv
from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool
from langchain.agents import create_agent


# Load environment variables
# Load the OpenAI API key from a .env file to authenticate with the OpenAI API.
load_dotenv(override=True)  

client = OpenAI()

search_model = init_chat_model("gpt-4.1-mini", model_provider="openai", temperature=0)
search = GoogleSerperAPIWrapper()

@tool
def intermediate_answer(query: str) -> str:
    """Useful for when you need to ask with search."""
    return search.run(query)

tools = [intermediate_answer]
agent = create_agent(search_model, tools)

# Step 2: Define state structure to track input, decision, and output
class State(TypedDict):
    input: str  
    decision: str  
    output: str  

class Route(BaseModel):
    step: Literal["general", "news"] = Field(None, description="The next step in the routing process")

# Function to determine the users query type using AI
def get_router_response(input_text: str) -> str:
    """Uses AI model to categorize input into a specific genre."""
    response = client.chat.completions.create(model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "Route the input to 'general' or 'news' based on whether the user is asking current events and news or general information. If unsure, default to 'general'."},
        {"role": "user", "content": input_text},
    ])
    query_type = response.choices[0].message.content.strip().lower()
    return query_type if query_type in ["general", "news"] else "general"

def handle_general_query(state: State):
    """Get LLM to handle general queries."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers general questions."},
            {"role": "user", "content": state['input']}
        ],
        max_tokens=500
    )
    return {"output": response.choices[0].message.content.strip(), "decision": "General"}

def handle_news_query(state: State):
    """Get LLM to handle news queries."""
    search = GoogleSerperAPIWrapper(type="news")
    results = search.results(state['input'])
    return {"output": f"News results for '{state['input']}': {results}", "decision": "News"}

# Routing function to determine which function to call
def route_request(state: State):
    """Determines the genre and routes accordingly."""
    decision = get_router_response(state["input"])
    return {"decision": decision}

def route_decision(state: State):
    """Maps the decision to the correct function."""
    decision = state["decision"].lower()
    if decision == "general":
        return "handle_general_query"
    elif decision == "news":
        return "handle_news_query"
    # Default to general query handler if decision is unrecognized
    return "handle_general_query"

# Build LangGraph workflow
def build_workflow():
    """Constructs the parallel workflow."""
    workflow = StateGraph(State)
    
    workflow.add_node("handle_general_query", handle_general_query)
    workflow.add_node("handle_news_query", handle_news_query)
    workflow.add_node("route_request", route_request)

    workflow.add_edge(START, "route_request")
    workflow.add_conditional_edges(
        "route_request",
        route_decision,
        {
            "handle_general_query": "handle_general_query",
            "handle_news_query": "handle_news_query",
        },
    )
    workflow.add_edge("handle_general_query", END)
    workflow.add_edge("handle_news_query", END)

    return workflow.compile()

# Implement the Streamlit UI
def run_streamlit_app():
    """Creates an interactive UI for general or news related queries."""
    st.title("NewsGenie: Your AI-Powered News Assistant")
    user_input = st.text_input("Enter your query", "")

    if st.button("Generate Response"):
        if user_input:
            workflow = build_workflow()
            state = workflow.invoke({"input": user_input})
            st.subheader("Detected query type:")
            st.write(state["decision"].capitalize())
            st.subheader("Generated Response:")
            st.write(state["output"])

if __name__ == "__main__":
    run_streamlit_app()