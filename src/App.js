import React, { useRef, useState, useEffect } from "react";
import axios from "axios";
import "./ChatPage.css";


function parseAssistantResponse(text) {
  // Each paper: paperX: title: ..., l2_distance: ..., explanation: ...
  const regex = /paper(\d+):\s*title:\s*(.*?),\s*cosine_similarity:\s*([\d.]+),\s*explanation:\s*(.*?)(?=(?:paper\d+:|$))/gs;
  let result, papers = [];
  while ((result = regex.exec(text)) !== null) {
    papers.push({
      paper: result[1],
      title: result[2],
      l2: result[3],
      explanation: result[4].trim()
    });
  }
  return papers.length ? papers : null;
}

function ChatMessage({ message }) {
  if (message.role === "assistant") {
    const papers = parseAssistantResponse(message.text);
    if (papers) {
      return (
        <div className="msg-row msg-assistant">
          <div className="msg-bubble" style={{padding:0}}>
            <div style={{overflowX:"auto"}}>
              <table className="resp-table">
                <thead>
                  <tr>
                    <th>Paper</th>
                    <th>Title</th>
                    <th>Cosine Similarity</th>
                    <th>Explanation</th>
                  </tr>
                </thead>
                <tbody>
                  {papers.map(p => (
                    <tr key={p.paper}>
                      <td>{p.paper}</td>
                      <td>{p.title}</td>
                      <td>{p.l2}</td>
                      <td style={{minWidth:140}}>{p.explanation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      );
    }
  }
  // Default message bubble
  return (
    <div className={`msg-row ${message.role === "user" ? "msg-user" : "msg-assistant"}`}>
      <div className="msg-bubble">
        {message.text}
      </div>
    </div>
  );
}


function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hello! Ask me a research question. Set k for how many papers to check." }
  ]);
  const [input, setInput] = useState("");
  const [kValue, setKValue] = useState(5);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const question = input.trim();
    setMessages(msgs => [...msgs, { role: "user", text: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await axios.post("http://143.110.237.142:8000/rag_cosine", {
        user_question: question,
        top_k: kValue,
      });
      setMessages(msgs => [
        ...msgs,
        { role: "assistant", text: res.data.output || "No answer." },
      ]);
    } catch (err) {
      setMessages(msgs => [
        ...msgs,
        { role: "assistant", text: "Error: " + (err.response?.data?.detail || err.message) },
      ]);
    }
    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-outer">
      <div className="chat-box">
        <div className="chat-header">Research Assistant Chat</div>
        <div className="chat-body" id="chat-body">
          {messages.map((msg, i) => <ChatMessage message={msg} key={i} />)}
          <div ref={chatEndRef} />
        </div>
        <div className="chat-input-area">
          <input
            type="number"
            min={1}
            max={20}
            value={kValue}
            onChange={e => setKValue(Number(e.target.value))}
            className="k-input"
            disabled={loading}
            title="Number of papers (k)"
          />
          <textarea
            className="msg-input"
            rows={1}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your research question..."
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            className="send-btn"
            disabled={loading || !input.trim()}
          >
            {loading ? "..." : "Send"}
          </button>
        </div>
      </div>
      <footer className="chat-footer">© 2025 Research Assistant Demo</footer>
    </div>
  );
}

export default App;
