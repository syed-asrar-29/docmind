import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

sys.path.insert(0, str(ROOT_DIR))

from ingestion.loader import index_documents
from agents.planner import plan_subquestions
from agents.rag_agent import answer_subquestion
from agents.synthesizer import synthesize_answer
from memory.store import check_memory, save_memory


st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")

load_dotenv(ROOT_DIR / ".env")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root {
  --bg: #0f1117;
  --sidebar: #1a1d27;
  --card: #1e2130;
  --text: #ffffff;
  --muted: #9aa3b2;
  --grad-start: #7c3aed;
  --grad-end: #2563eb;
  --border: #2a2f3e;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--bg);
  color: var(--text);
}

.stApp {
  background: radial-gradient(circle at 20% 20%, #1a1f2e 0%, #0f1117 55%);
}

[data-testid="stSidebar"] {
  background-color: var(--sidebar);
  border-right: 1px solid var(--border);
}

.sidebar-logo {
  font-size: 28px;
  font-weight: 700;
  background: linear-gradient(90deg, var(--grad-start), var(--grad-end));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.sidebar-subtitle {
  color: var(--muted);
  margin-top: 4px;
  font-size: 13px;
}

.sidebar-section {
  font-size: 14px;
  font-weight: 600;
  margin: 14px 0 8px 0;
}

.sidebar-divider {
  height: 1px;
  background: var(--border);
  margin: 14px 0;
}

[data-testid="stFileUploader"] {
  border: 2px dashed #7c3aed;
  border-radius: 14px;
  padding: 12px;
  transition: box-shadow 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
  box-shadow: 0 0 18px rgba(124, 58, 237, 0.4);
}

.file-pill {
  display: inline-flex;
  align-items: center;
  padding: 6px 12px;
  margin: 4px 6px 4px 0;
  border-radius: 999px;
  background: rgba(124, 58, 237, 0.2);
  color: #c4b5fd;
  font-size: 12px;
}

.main-title {
  font-size: 40px;
  font-weight: 700;
  background: linear-gradient(90deg, var(--grad-start), var(--grad-end));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0;
}

.main-subtitle {
  color: var(--muted);
  margin-top: 2px;
  font-size: 15px;
}

.chat-container {
  height: 65vh;
  overflow-y: auto;
  padding-right: 10px;
}

.msg-row {
  display: flex;
  width: 100%;
  margin-bottom: 12px;
}

.msg-row.user {
  justify-content: flex-end;
}

.msg-row.bot {
  justify-content: flex-start;
}

.msg {
  max-width: 70%;
  padding: 14px 16px;
  border-radius: 18px;
  position: relative;
  animation: fadeIn 0.3s ease-in;
}

.msg.user {
  background: linear-gradient(120deg, var(--grad-start), var(--grad-end));
  color: #ffffff;
  text-align: left;
}

.msg.bot {
  background: var(--card);
  color: #ffffff;
  border-left: 3px solid var(--grad-start);
}

.msg-meta {
  font-size: 11px;
  color: #cbd5f5;
  margin-top: 8px;
  display: flex;
  justify-content: space-between;
}

.msg-meta .tag {
  font-size: 10px;
  color: #d1d5ff;
  background: rgba(124, 58, 237, 0.2);
  padding: 2px 6px;
  border-radius: 999px;
}

.sources {
  margin: 6px 0 12px 0;
  font-size: 12px;
}

.sources summary {
  cursor: pointer;
  color: #a78bfa;
}

.source-badge {
  display: inline-block;
  margin: 6px 6px 0 0;
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.2);
  color: #bfdbfe;
  font-size: 11px;
}

.typing {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  border-radius: 18px;
  background: var(--card);
  border-left: 3px solid var(--grad-start);
  animation: fadeIn 0.2s ease-in;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #c4b5fd;
  animation: bounce 1s infinite;
}

.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

.input-wrap {
  position: sticky;
  bottom: 0;
  background: rgba(15, 17, 23, 0.9);
  padding: 12px 0;
  border-top: 1px solid var(--border);
}

.stTextInput > div > div > input {
  background: #141824;
  border: 1px solid #2a2f3e;
  color: #ffffff;
  border-radius: 12px;
}

.stTextInput > div > div > input:focus {
  border-color: #7c3aed;
  box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.3);
}

.stButton > button {
  border-radius: 12px;
  border: 1px solid #2a2f3e;
  background: #141824;
  color: #ffffff;
}

.stButton > button:hover {
  border-color: #7c3aed;
  color: #e9d5ff;
}

.empty-state {
  margin-top: 18vh;
  text-align: center;
  color: var(--muted);
}

.empty-state .arrow {
  font-size: 26px;
  margin-top: 8px;
  color: #7c3aed;
}

@keyframes bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-6px); }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def now_time() -> str:
    return datetime.now().strftime("%H:%M")


def render_message(message: dict) -> str:
    role = message["role"]
    content = message["content"]
    timestamp = message["timestamp"]

    if role == "user":
        return (
            "<div class='msg-row user'>"
            "<div class='msg user'>"
            f"<div>{content}</div>"
            f"<div class='msg-meta'><span></span><span>{timestamp}</span></div>"
            "</div></div>"
        )

    tag = "Powered by Llama 3.1"
    return (
        "<div class='msg-row bot'>"
        "<div class='msg bot'>"
        f"<div>{content}</div>"
        f"<div class='msg-meta'><span>{timestamp}</span><span class='tag'>{tag}</span></div>"
        "</div></div>"
    )


def render_sources(sources: list[str]) -> str:
    if not sources:
        return ""

    badges = "".join([f"<span class='source-badge'>{src}</span>" for src in sources])
    return (
        "<details class='sources'>"
        "<summary>Sources</summary>"
        f"<div>{badges}</div>"
        "</details>"
    )


def save_uploaded_files(uploaded_files: list) -> list[str]:
    if not uploaded_files:
        return []

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in uploaded_files:
        target = DATA_DIR / file.name
        with target.open("wb") as handle:
            handle.write(file.getbuffer())
        saved.append(file.name)

    return saved


def get_existing_files() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted([p.name for p in DATA_DIR.iterdir() if p.is_file()])


def handle_submit(question: str, top_k: int, use_memory: bool) -> dict:
    if not question.strip():
        return {"answer": "", "sources": []}

    if not get_existing_files():
        return {
            "answer": "No documents loaded yet. Upload files in the sidebar to begin.",
            "sources": [],
        }

    if not os.getenv("GROQ_API_KEY"):
        return {
            "answer": "GROQ_API_KEY is missing. Add it to your .env file and restart.",
            "sources": [],
        }

    if use_memory:
        cached = check_memory(question)
        if cached:
            return {"answer": cached, "sources": []}

    subquestions = plan_subquestions(question)
    subanswers = [answer_subquestion(q, top_k=top_k) for q in subquestions]
    synthesis = synthesize_answer(question, subanswers)

    answer = synthesis["final_answer"]
    sources = synthesis["all_sources"]

    if use_memory:
        save_memory(question, answer)

    return {"answer": answer, "sources": sources}


if "messages" not in st.session_state:
    st.session_state.messages = []

if "submitted" not in st.session_state:
    st.session_state.submitted = False


st.sidebar.markdown(
    "<div class='sidebar-logo'>DocMind</div>", unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div class='sidebar-subtitle'>Multi-Agent RAG Assistant</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='sidebar-section'>Upload Documents</div>", unsafe_allow_html=True
)

uploaded_files = st.sidebar.file_uploader(
    "Drop files here",
    type=["pdf", "csv", "json", "txt"],
    accept_multiple_files=True,
    label_visibility="visible",
)

indexed_files = st.session_state.get("indexed_files", set())

if uploaded_files:
    saved_names = save_uploaded_files(uploaded_files)
    new_names = [name for name in saved_names if name not in indexed_files]
    if new_names:
        index_documents(DATA_DIR)
        indexed_files.update(new_names)
        st.session_state.indexed_files = indexed_files
        st.toast("Documents indexed successfully", icon="✅")

existing_files = get_existing_files()
if existing_files:
    pills = "".join([f"<span class='file-pill'>{name}</span>" for name in existing_files])
    st.sidebar.markdown(pills, unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-section'>Settings</div>", unsafe_allow_html=True)

source_count = st.sidebar.slider("Number of sources", 1, 8, 4)
use_memory_cache = st.sidebar.toggle("Use memory cache", value=True)

st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='sidebar-subtitle'>Powered by Groq + LangChain</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>DocMind</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Ask anything about your documents</div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Summarize my documents"):
            st.session_state.user_input = "Summarize my documents"
            st.session_state.submitted = True
    with col2:
        if st.button("Find key insights"):
            st.session_state.user_input = "Find key insights"
            st.session_state.submitted = True
    with col3:
        if st.button("List main topics"):
            st.session_state.user_input = "List main topics"
            st.session_state.submitted = True

chat_container = st.container()
with chat_container:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        st.markdown(render_message(message), unsafe_allow_html=True)
        if message["role"] == "assistant":
            st.markdown(render_sources(message.get("sources", [])), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if not existing_files:
    st.markdown(
        "<div class='empty-state'>No documents loaded yet" "<div class='arrow'>↖</div></div>",
        unsafe_allow_html=True,
    )


def mark_submitted() -> None:
    st.session_state.submitted = True


with st.container():
    st.markdown("<div class='input-wrap'>", unsafe_allow_html=True)
    input_cols = st.columns([8, 1.2, 1.2])
    with input_cols[0]:
        st.text_input(
            "",
            key="user_input",
            placeholder="Ask DocMind anything...",
            on_change=mark_submitted,
            label_visibility="collapsed",
        )
    with input_cols[1]:
        send_clicked = st.button("Send")
    with input_cols[2]:
        clear_clicked = st.button("Clear")
    st.markdown("</div>", unsafe_allow_html=True)

if clear_clicked:
    st.session_state.messages = []
    st.session_state.submitted = False
    st.rerun()
    st.experimental_rerun()

if send_clicked or st.session_state.submitted:
    question = st.session_state.get("user_input", "").strip()
    st.session_state.submitted = False
    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question, "timestamp": now_time()}
        )

        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            "<div class='typing'><span class='dot'></span><span class='dot'></span><span class='dot'></span></div>",
            unsafe_allow_html=True,
        )

        result = handle_submit(question, source_count, use_memory_cache)
        typing_placeholder.empty()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "timestamp": now_time(),
                "sources": result["sources"],
            }
        )

        st.session_state.submitted = False
        st.rerun()
