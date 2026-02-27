from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import pprint

# Load environment variables
load_dotenv(override=True) 


def output_result(result: str) -> str:
    if not result['messages']:
        print("No messages found in the result.")
        return
    
    for message in result['messages']:
        if isinstance(message, ToolMessage):
            print("Tool message content:")
            pprint.pprint(message.content, indent=4)
        elif isinstance(message, AIMessage):
            print("AI message content:")
            pprint.pprint(message.content, indent=4)
        elif isinstance(message, HumanMessage):
            continue # Skip human messages
    


def main():
    print("lanchain_tavily search")

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    tools = [TavilySearch()]
    agent = create_agent(model=llm, 
                         tools=tools)
    
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="What is the latest news on AI?"
            )
        }
    )
    
    output_result(result)


if __name__ == "__main__":    
    main()