"""Streamlit interface:  python -m streamlit run app.py"""
import re
from pathlib import Path

import streamlit as st

from rag.agent import AgenticRAG, Result, StandardRAG
from rag.config import UPLOADS_DIR, groq_api_key
from rag.ingest.extract import SUPPORTED
from rag.llm import GroqLLM, LLMError
from rag.retrieval import MODES, KnowledgeBase

st.set_page_config(page_title="Agentic RAG", page_icon=":material/search_insights:", layout="wide")
# Each paragraph picks its own direction, so Arabic answers read right to left.
st.markdown("""<style>
[data-testid="stMarkdownContainer"] :is(p, li, h1, h2, h3, h4, td, th) {unicode-bidi: plaintext; text-align: start;}
[data-testid="stChatInput"] textarea {unicode-bidi: plaintext;}
</style>""", unsafe_allow_html=True)

EXAMPLES = [
    "How did operating profit change from 2024 to 2025, in EGP million and percent?",
    "Compare the 2023 and 2026 remote work policies and list what changed.",
    "Did the company meet the three goals it set for 2025 in the 2024 annual report?",
    "Which department has the highest average salary, and what bonus would a score of 4.0 pay on that average?",
    "ما هي تكلفة Nilora Care لجهاز Breeze Smart AC بسعر 2025، وكم تصبح مدة الضمان؟",
]
TOOL_LABELS = {"search_knowledge_base": "Search", "compare_documents": "Compare documents", "analyze_data": "Analyze data",
               "summarize_document": "Summarize", "calculate": "Calculate", "list_documents": "List documents"}
STRATEGY = {"direct": "Direct answer", "single": "Single lookup", "multi_step": "Multi-step research"}


@st.cache_resource(show_spinner="Loading the knowledge base (the first run downloads a 120 MB embedding model)")
def get_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.refresh()
    return kb


def api_key() -> str:
    try:
        secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file
        secret = ""
    return groq_api_key() or secret or st.session_state.get("typed_key", "").strip()


def md(text: str) -> str:
    return text.replace("$", r"\$")  # otherwise Streamlit renders $...$ as LaTeX


def args_text(args: dict) -> str:
    parts = []
    for key, value in args.items():
        parts.append(", ".join(map(str, value)) if isinstance(value, list) else str(value))
    return " | ".join(p for p in parts if p)


# --- rendering ---------------------------------------------------------------
def show_event(event) -> str | None:
    """Draw one progress event; return a status label."""
    data = event.data
    if event.type == "plan":
        st.markdown(md(f"**Plan:** {STRATEGY[data['strategy']]}. {data['reason']}"))
        for step in data["steps"] or data["sub_questions"]:
            st.markdown(md(f"- {step}"))
        return "Researching"
    if event.type == "tool":
        st.markdown(f"**{TOOL_LABELS.get(data['name'], data['name'])}**")
        st.code(args_text(data["args"]) or "-", language="sql" if data["name"] == "analyze_data" else None,
                wrap_lines=True)
        with st.expander("Result"):
            st.text(data["output"])
        return "Researching"
    if event.type == "verify":
        if data["ok"]:
            st.markdown("**Check:** answer is supported by its sources." if data["checked"] else
                        "**Check:** skipped, the reviewer reply was unreadable.")
        else:
            st.markdown(md("**Check found problems:** " + "; ".join(data["issues"])))
        return "Checking the answer"
    if event.type == "revise":
        st.markdown("**Revising the answer.**")
        return "Revising"
    return None


def show_result(result: Result) -> None:
    st.markdown(md(result.answer))
    if result.verification and not result.verification.get("ok", True):
        st.warning("The reviewer still had doubts: " + md("; ".join(result.verification["issues"])))
    if result.sources:
        with st.expander(f"Sources ({len(result.sources)})"):
            for s in result.sources:
                st.markdown(md(f"**[{s.number}] {s.label}**"))
                st.text(s.text)
    facts = [f"{result.seconds:.1f} s", f"{result.tokens:,} tokens"]
    if result.plan:
        facts.insert(0, f"{result.tool_calls} tool calls")
    st.caption(" · ".join(facts + result.models))


def run_agent(question: str, history, verify: bool) -> tuple[Result, list]:
    events = []
    with st.status("Planning", expanded=True) as status:
        for event in AgenticRAG(get_kb(), GroqLLM(api_key()), verify).run(question, history):
            if event.type == "answer":
                status.update(label="Reasoning steps", state="complete", expanded=False)
                return event.data["result"], events
            events.append(event)
            status.update(label=show_event(event), expanded=True)
    raise LLMError("The agent stopped without an answer.")


# --- views -------------------------------------------------------------------
def assistant_view(verify: bool) -> None:
    st.title("Document assistant")
    st.caption("Ask in English or Arabic. The agent plans, picks its tools, checks its answer and cites sources.")
    turns = st.session_state.setdefault("turns", [])
    for turn in turns:
        st.chat_message("user").markdown(md(turn["question"]))
        with st.chat_message("assistant"):
            with st.status("Reasoning steps", state="complete", expanded=False):
                for event in turn["events"]:
                    show_event(event)
            show_result(turn["result"])
    question = st.chat_input("Ask about the documents") or st.session_state.pop("pending", None)
    slot = st.empty()  # cleared at once when a question starts, so no faded examples stay on screen
    if not turns and not question:
        with slot.container():
            st.markdown("**Try one of these**")
            for example in EXAMPLES:
                if st.button(example):
                    st.session_state["pending"] = example
                    st.rerun()
    if not question:
        return
    st.chat_message("user").markdown(md(question))
    with st.chat_message("assistant"):
        try:
            result, events = run_agent(question, [(t["question"], t["result"].answer) for t in turns], verify)
        except LLMError as e:
            st.error(str(e))
            return
        show_result(result)
    turns.append({"question": question, "result": result, "events": events})


def compare_view(verify: bool) -> None:
    st.title("Standard RAG vs Agentic RAG")
    st.caption("The same question through both systems. Standard RAG does one semantic search and one answer. "
               "The agent plans, uses several tools and checks itself.")
    choice = st.selectbox("Example", EXAMPLES, index=2)
    question = st.text_input("Question", choice)
    if not st.button("Run both", type="primary") or not question.strip():
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Standard RAG")
        try:
            with st.spinner("Answering"):
                standard = StandardRAG(get_kb(), GroqLLM(api_key())).ask(question)
            show_result(standard)
        except LLMError as e:
            st.error(str(e))
    with right:
        st.subheader("Agentic RAG")
        try:
            result, _ = run_agent(question, [], verify)
            show_result(result)
        except LLMError as e:
            st.error(str(e))


def knowledge_view() -> None:
    kb = get_kb()
    st.title("Knowledge base")
    totals = st.container()  # filled below, after any upload has been indexed

    st.subheader("Add documents")
    uploads = st.file_uploader("PDF, Word, Markdown, text, CSV or Excel", accept_multiple_files=True,
                               type=sorted(s[1:] for s in SUPPORTED))
    if uploads and st.button("Add to knowledge base", type="primary"):
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            name = re.sub(r"[^\w.\- ]", "_", Path(f.name).name)
            if name in kb.docs and not (UPLOADS_DIR / name).exists():
                name = f"upload_{name}"
            (UPLOADS_DIR / name).write_bytes(f.getvalue())
        with st.spinner("Indexing"):
            report = kb.refresh()
        for doc, error in report["failed"].items():
            (UPLOADS_DIR / doc).unlink(missing_ok=True)  # do not keep files that cannot be read
            st.error(f"{doc}: {error}")
        if report["added"]:
            st.success("Added " + ", ".join(report["added"]))

    a, b, c = totals.columns(3)
    a.metric("Documents", len(kb.docs))
    b.metric("Chunks", len(kb.chunks))
    c.metric("Data tables", len(kb.tables))

    st.subheader("Documents")
    for doc, title in kb.docs.items():
        chunks = kb.document_chunks(doc)
        with st.expander(f"{doc} · {title} · {len(chunks)} chunks"):
            if (UPLOADS_DIR / doc).exists() and st.button("Remove", key=f"rm_{doc}"):
                (UPLOADS_DIR / doc).unlink()
                kb.refresh()
                st.rerun()
            for ch in chunks:
                st.markdown(md(f"**{ch.section}**" + (f" (page {ch.page})" if ch.page else "")))
                st.text(ch.text)

    st.subheader("Try the retriever")
    query = st.text_input("Search query")
    mode = st.segmented_control("Mode", MODES, default="hybrid")
    if query:
        for ch in kb.search(query, mode or "hybrid", 5):
            st.markdown(md(f"**{ch.label}**"))
            st.text(ch.text)


# --- page --------------------------------------------------------------------
with st.sidebar:
    view = st.radio("View", ["Assistant", "Standard vs Agentic", "Knowledge base"], label_visibility="collapsed")
    st.divider()
    verify_on = st.toggle("Check answers before showing", True, help="One extra call that compares the answer with its "
                          "sources and triggers a revision when something is unsupported.")
    if st.button("Clear conversation"):
        st.session_state["turns"] = []
    if not api_key() or st.session_state.get("typed_key"):
        st.text_input("Groq API key", type="password", key="typed_key",
                      help="Free at console.groq.com/keys. Kept for this session only. Or put it in .env.")
        if not api_key():
            st.info("Add a Groq API key to ask questions.")

if view == "Knowledge base":
    knowledge_view()
elif not api_key():
    st.title("Agentic RAG")
    st.write("Add your Groq API key in the sidebar, or in the .env file, to start asking questions. "
             "The knowledge base view works without a key.")
elif view == "Assistant":
    assistant_view(verify_on)
else:
    compare_view(verify_on)
