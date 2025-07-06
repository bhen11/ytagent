import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch  # Updated import
import gradio as gr

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = "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"

# LangGraph State for QA
class QAState(TypedDict):
    video_url: str
    question: str
    transcript: str
    relevant_chunks: List[str]
    answer: str
    chat_history: List[str]

# Transcript extraction
transcript_cache = {}  # cache to avoid repeat fetch

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

# def web_fallback(state: QAState) -> QAState:
#     question = state["question"]
#     tool = TavilySearch(tavily_api_key=TAVILY_API_KEY)
#     response = tool.invoke({"query": question})

#     results = response.get("results", [])
#     web_answer = "\n\n".join([res.get("content", "") for res in results]) if results else "No web answer available."

#     new_history = state.get("chat_history", []) + [f"Q: {question}\nA: {web_answer}"]
#     return {**state, "answer": web_answer, "chat_history": new_history}
def web_fallback(state: QAState) -> QAState:
    question = state["question"]
    tool = TavilySearch(tavily_api_key=TAVILY_API_KEY)

    try:
        response = tool.invoke({"query": question})
        results = response.get("results", [])

        if isinstance(results, list) and results:
            # Use top 1–3 contents
            contents = [res.get("content", "") for res in results[:3] if "content" in res]
            web_answer = "\n\n".join(contents).strip() or "No useful content in search results."
        else:
            web_answer = "No web results found."

    except Exception as e:
        web_answer = f"Web fallback error: {str(e)}"

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

# Gradio UI setup
chat_history = []
transcript_text = ""

def handle_summary(video_url):
    global transcript_text
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL"
    if video_id in transcript_cache:
        transcript = transcript_cache[video_id]
    else:
        transcript = " ".join([seg["text"] for seg in YouTubeTranscriptApi.get_transcript(video_id)])
        transcript_cache[video_id] = transcript
    transcript_text = transcript
    return summarize_transcript(transcript)

def summarize_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)
    chunk_text = "\n".join(chunks)
    prompt = f"""
Summarize the following transcript from a YouTube video:

{chunk_text}

Summary:
"""
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")
    result = llm.invoke(prompt)
    return result.content.strip()

def handle_question(video_url, question):
    global chat_history
    state = qa_graph.invoke({
        "video_url": video_url,
        "question": question,
        "chat_history": chat_history
    })
    chat_history = state["chat_history"]
    return state["answer"], "\n\n".join(chat_history)

with gr.Blocks() as demo:
    gr.Markdown("# 🎥 YouTube Video Summarizer + Chatbot with Memory + Web Fallback")

    video_url = gr.Textbox(label="YouTube URL")

    with gr.Row():
        summary_btn = gr.Button("🟢 Generate Summary")
        summary_output = gr.Textbox(label="📄 Summary")

    with gr.Row():
        question_input = gr.Textbox(label="❓ Ask a Question")
        answer_output = gr.Textbox(label="💬 Answer")

    chat_log = gr.Textbox(label="📜 Chat History", lines=10)

    summary_btn.click(fn=handle_summary, inputs=video_url, outputs=summary_output)
    question_input.submit(fn=handle_question, inputs=[video_url, question_input], outputs=[answer_output, chat_log])

if __name__ == "__main__":
    demo.launch()
