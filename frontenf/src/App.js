import React, { useState } from "react";
import axios from "axios";
import TextareaAutosize from "react-textarea-autosize";
import "./App.css";

function App() {
  const [videoUrl, setVideoUrl] = useState("");
  const [summary, setSummary] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [chatHistory, setChatHistory] = useState([]);

  const handleSummary = async () => {
    try {
      const res = await axios.post("http://localhost:8000/summary", {
        video_url: videoUrl,
      });
      setSummary(res.data.summary);
    } catch (error) {
      console.error("Error fetching summary:", error);
    }
  };

  const handleQuestion = async () => {
    try {
      const res = await axios.post("http://localhost:8000/question", {
        video_url: videoUrl,
        question,
        chat_history: chatHistory,
      });
      setAnswer(res.data.answer);
      setChatHistory(res.data.chat_history);
    } catch (error) {
      console.error("Error fetching answer:", error);
    }
  };

  return (
    <div className="container">
      <h2>YouTube Summarizer + Chatbot</h2>

      <div className="section">
        <input
          type="text"
          placeholder="Enter YouTube URL"
          value={videoUrl}
          onChange={(e) => setVideoUrl(e.target.value)}
        />
        <button onClick={handleSummary}>Generate Summary</button>
      </div>

      <div className="section">
        <h4>Summary:</h4>
        <TextareaAutosize
          value={summary}
          readOnly
          className="autosize-textarea"
        />
      </div>

      <div className="section">
        <input
          type="text"
          placeholder="Ask a question about the video"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={handleQuestion}>Ask</button>
      </div>

      <div className="section">
        <h4>Answer:</h4>
        <TextareaAutosize
          value={answer}
          readOnly
          className="autosize-textarea"
        />
      </div>

      <div className="section">
        <h4>Chat History:</h4>
        <TextareaAutosize
          value={chatHistory.join("\n\n")}
          readOnly
          className="autosize-textarea"
        />
      </div>
    </div>
  );
}

export default App;
