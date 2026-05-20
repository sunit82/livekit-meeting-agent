# graph.py
from dotenv import load_dotenv
import os
import shutil
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

from model_factory import ModelFactory

load_dotenv(".env.local")

# -------------------- Build your Meeting RAG pipeline --------------------
def create_workflow():
    model_provider = ModelFactory.create()
    llm = model_provider.fetch_llm()
    embeddings = model_provider.fetch_embeddings()

    pdf_path = os.getenv("COMPANY_PDF_PATH", "./CompanyInfo.pdf")
    if not os.path.exists(pdf_path):
        print(f"No Company Info PDF found. No company information will be available.")
        # raise FileNotFoundError(f"PDF file not found: {pdf_path}. Please set COMPANY_PDF_PATH environment variable or place TechCompanyInfo.pdf in the current directory.")
        pages = []

    else:
        pages = PyPDFLoader(pdf_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    persist_directory = os.getenv("CHROMA_DIR", "./chroma_store")
    shutil.rmtree(persist_directory, ignore_errors=True)
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="company_info",
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    @tool
    def company_info_tool(query: str) -> str:
        """Searches the company information document and returns relevant chunks about the company."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the company documents."
        # Build the result string step by step for beginners
        result_parts = []
        for i, doc in enumerate(docs):
            info_number = i + 1
            content = doc.page_content
            formatted_info = f"Info {info_number}:\n{content}"
            result_parts.append(formatted_info)
        
        # Join all parts with double newlines
        return "\n\n".join(result_parts)

    @tool
    def record_answer_tool(answer: str) -> str:
        """Records the candidate's answer to a text file for later review."""
        # Simply write the answer to the file
        with open("interview_answers.txt", "a", encoding="utf-8") as f:
            f.write(f"\nAnswer:\n{answer}\n")
            f.write("-" * 50 + "\n")
        
        print(f"Recorded answer: {answer[:50]}...")
        return "Track updates recorded successfully!"

    tools = [company_info_tool, record_answer_tool]
    llm = llm.bind_tools(tools)

    class MeetingState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def decide_next_action(state: MeetingState) -> str:
        """Decide what to do next: tool_executor or end"""
        last = state["messages"][-1]
        
        # Check if we need to execute tool calls from the LLM
        if hasattr(last, "tool_calls") and last.tool_calls and len(last.tool_calls) > 0:
            return "tool_executor"
        
        # Default to end if no specific action needed
        return "end"

    def call_llm(state: MeetingState) -> MeetingState:
        """Main LLM call that handles the Meeting conversation."""
        system_prompt = (
            "You are a professional personal assistant called Claudia."
            "You will wait for the host to ask you to start the fortnightly meeting and then proceed with the agenda items one by one as per the graph. "
            "If asked for self introduction, you will introduce yourself as Claudia, a software project manager  personal assistant. You are powered by LangGraph, LiveKit and Ollama. You are here to assist in running the fortnightly meeting."
            "When asked to start the meeting, first thank Sunit for inviting you to the meeting and then proceed with the agenda items one by one as per the graph. "
            "You will run a agenda driven meething in this order:\n"
            "You will ask the track owner only once to share the track updates as per agenda and then wait for them to complete their updates. Do not ask them again to share the updates. Wait for the track owner to complete their updates and till someone explicitly asks to continue to the next agenda topic. "
            "As per stated 10 point agenda below, as owner of the meething  you will call upon each track owner to share their track updates."
            "You will wait for the track owner to complete the updates and till someone explicitly asks to continue to the next agenda topic."
            # "Agenda of the meeting in bulleted points will go here followed by some IMPORTANT ROUTING RULES if any\n"
        )
        
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm.invoke(msgs)
        return {"messages": [message]}

    def tool_executor(state: MeetingState) -> MeetingState:
        """Execute tool calls from the LLM's response."""
        # Get the tool calls from the last message
        tool_calls = state["messages"][-1].tool_calls
        results = []

        # Go through each tool call one by one
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            
            print(f"Running tool: {tool_name}")

            # Call the right tool based on its name
            if tool_name == "company_info_tool":
                result = company_info_tool.invoke(tool_args)
            elif tool_name == "record_answer_tool":
                result = record_answer_tool.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            # Create a message with the result
            tool_message = ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_name,
                content=str(result),
            )
            results.append(tool_message)

        print("All tools finished running.")
        return {"messages": results}


    # Build the Meeting graph
    graph = StateGraph(MeetingState)
    
    # Add nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tool_executor", tool_executor)
    
    # Set up the flow
    graph.set_entry_point("llm")
    
    # Conditional edges from LLM
    graph.add_conditional_edges(
        "llm", 
        decide_next_action, 
        {
            "tool_executor": "tool_executor",
            "end": END
        }
    )
    
    # From tool_executor back to LLM
    graph.add_edge("tool_executor", "llm")

    return graph.compile()