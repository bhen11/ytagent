import os
import gradio as gr
from langgraph.graph import StateGraph
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

# Node 1: Fetch transcript
def fetch_transcript_node(state):
    video_url = state["video_url"]
    # Extract video ID from various YouTube URL formats
    if "youtu.be" in video_url:
        video_id = video_url.split("/")[-1].split("?")[0]
    elif "youtube.com/watch?v=" in video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
    else:
        raise ValueError("Invalid YouTube URL format")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([seg["text"] for seg in transcript])
    return {"transcript": full_text}

# Node 2: Chunk transcript
def chunk_transcript_node(state):
    text = state["transcript"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return {"chunks": chunks}

# Node 3: Summarize chunks
def summarize_chunks_node(state):
    summaries = []
    for chunk in state["chunks"]:
        prompt = f"Summarize the following YouTube transcript chunk:\n\n{chunk}"
        res = llm([HumanMessage(content=prompt)])
        summaries.append(res.content)
    return {"summaries": summaries}

# Node 4: Combine all summaries
def combine_summaries_node(state):
    summaries = state["summaries"]
    final_prompt = "Combine these summaries into a short, clear summary:\n\n" + "\n\n".join(summaries)
    result = llm([HumanMessage(content=final_prompt)])
    return {"final_summary": result.content}

# LangGraph pipeline
def build_graph():
    builder = StateGraph(dict)
    builder.add_node("fetch", fetch_transcript_node)
    builder.add_node("chunk", chunk_transcript_node)
    builder.add_node("summarize", summarize_chunks_node)
    builder.add_node("combine", combine_summaries_node)
    builder.set_entry_point("fetch")
    builder.add_edge("fetch", "chunk")
    builder.add_edge("chunk", "summarize")
    builder.add_edge("summarize", "combine")
    return builder.compile()

# Gradio interface
graph = build_graph()
def run_pipeline(video_url):
    state = {"video_url": video_url}
    result = graph.invoke(state)
    return result["final_summary"]

gr.Interface(
    fn=run_pipeline,
    inputs=gr.Textbox(label="Enter YouTube URL"),
    outputs=gr.Textbox(label="Summary", lines=20),
    title="YouTube Video Summarizer",
).launch()
