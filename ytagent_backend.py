import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangGraph state type
class QAState(TypedDict):
    video_url: str
    question: str
    transcript: str
    relevant_chunks: List[str]
    answer: str
    chat_history: List[str]

transcript_cache = {}

def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com/watch?v=" in url:
        return url.split("v=")[-1].split("&")[0]
    return None

def fetch_transcript_node(state: QAState) -> QAState:
    video_url = state["video_url"]
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    if video_id in transcript_cache:
        full_text = transcript_cache[video_id]
    else:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([seg["text"] for seg in transcript])
        transcript_cache[video_id] = full_text

    return {**state, "transcript": full_text}

def chunk_transcript(state: QAState) -> QAState:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(state["transcript"])
    #extra 
    chunks = chunks[:10]
    return {**state, "relevant_chunks": chunks}

def answer_question(state: QAState) -> QAState:
    chunks = "\n".join(state["relevant_chunks"])
    question = state["question"]
    history = "\n".join(state.get("chat_history", []))
    prompt = f"""
You are a helpful assistant answering questions about a YouTube video transcript.
Only use the context below. If the answer isn't in the context, say "Answer not found in the video."

Context:
{chunks}

Chat History:
{history}

Question: {question}
Answer:
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    response = llm.invoke(prompt)
    answer = response.content.strip()
    new_history = state.get("chat_history", []) + [f"Q: {question}\nA: {answer}"]
    return {**state, "answer": answer, "chat_history": new_history}

def web_fallback(state: QAState) -> QAState:
    question = state["question"]
    tool = TavilySearch(tavily_api_key=TAVILY_API_KEY)
    results = tool.invoke({"query": question})

    if isinstance(results, str):
        web_answer = results
    elif isinstance(results, list):
        web_answer = "\n".join([
            res.get("content", "") if isinstance(res, dict) else str(res) 
            for res in results
        ])
    else:
        web_answer = str(results)

    new_history = state.get("chat_history", []) + [f"Q: {question}\nA: {web_answer}"]
    return {**state, "answer": web_answer, "chat_history": new_history}

# LangGraph setup
builder = StateGraph(QAState)
builder.add_node("fetch", fetch_transcript_node)
builder.add_node("chunk", chunk_transcript)
builder.add_node("answer", answer_question)
builder.add_node("fallback", web_fallback)

builder.set_entry_point("fetch")
builder.add_edge("fetch", "chunk")
builder.add_edge("chunk", "answer")
builder.add_conditional_edges("answer", lambda state: "fallback" if state["answer"].lower().strip() == "answer not found in the video." else END)
builder.add_edge("fallback", END)

qa_graph = builder.compile()

# Pydantic request models
class SummaryRequest(BaseModel):
    video_url: str

class QuestionRequest(BaseModel):
    video_url: str
    question: str
    chat_history: List[str] = []

@app.post("/summary")
def generate_summary(req: SummaryRequest):
    video_id = extract_video_id(req.video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}
    if video_id in transcript_cache:
        transcript = transcript_cache[video_id]
    else:
        transcript = " ".join([seg["text"] for seg in YouTubeTranscriptApi.get_transcript(video_id)])
        transcript_cache[video_id] = transcript

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)
    #extra
    chunks = chunks[:10]
    chunk_text = "\n".join(chunks)
    prompt = f"""
Summarize the following transcript from a YouTube video:

{chunk_text}

Summary:
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    result = llm.invoke(prompt)
    return {"summary": result.content.strip()}

@app.post("/question")
def ask_question(req: QuestionRequest):
    state = qa_graph.invoke({
        "video_url": req.video_url,
        "question": req.question,
        "chat_history": req.chat_history
    })
    return {
        "answer": state["answer"],
        "chat_history": state["chat_history"]
    }
