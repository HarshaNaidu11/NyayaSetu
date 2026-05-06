import streamlit as st
import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Nyaya Setu - Legal Aid",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
.confidence-high { background:#e1f5ee; color:#085041; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
.confidence-low { background:#fcebeb; color:#791f1f; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:500; }
.source-card { background:#f7f7f5; border-left:3px solid #7F77DD; padding:10px 14px; margin:6px 0; border-radius:0 8px 8px 0; font-size:13px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading legal database... please wait")
def load_pipeline():
    index = faiss.read_index("embeddings/legal_index.faiss")
    with open("embeddings/chunks.pkl", "rb") as f:
        store = pickle.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, store["texts"], store["metadatas"], embed_model


def retrieve(query, index, texts, metadatas, embed_model, k=5):
    query_vec = embed_model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    scores, indices = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "text": texts[idx],
                "metadata": metadatas[idx],
                "score": float(score)
            })
    return results


def translate_text(text, target="en"):
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


def detect_lang(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def generate_answer(query, chunks, api_key, lang_code="en"):
    context = "\n\n".join([
        f"Source {i+1} [{c['metadata']['title']}]:\n{c['text'][:300]}"
        for i, c in enumerate(chunks[:2])
    ])

    lang_instruction = ""
    if lang_code == "te":
        lang_instruction = "Answer in Telugu language."
    elif lang_code == "hi":
        lang_instruction = "Answer in Hindi language."

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{
            "role": "user",
            "content": f"""You are a legal helper for India. {lang_instruction}

Legal sources:
{context}

Question: {query}

Rules:
- Write 5 short sentences only
- Name the exact Indian law
- Last line: Go to [place] for free help
- Never mention US law

Answer:"""
        }],
        max_tokens=300,
    )
    return response.choices[0].message.content


# Pre-load pipeline at startup
with st.spinner("Loading legal database..."):
    try:
        index, texts, metadatas, embed_model = load_pipeline()
        pipeline_loaded = True
    except Exception as e:
        st.error(f"Failed to load database: {e}")
        pipeline_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Nyaya Setu")
    st.caption("Bridge to Justice for Every Indian")
    st.divider()

    api_key = st.text_input(
        "Groq API Key (free)",
        type="password",
        help="Get free key at console.groq.com"
    )

    language = st.selectbox(
        "Language / భాష / भाषा",
        ["English", "Telugu (తెలుగు)", "Hindi (हिन्दी)"]
    )
    lang_code = {
        "English": "en",
        "Telugu (తెలుగు)": "te",
        "Hindi (हिन्दी)": "hi"
    }[language]

    st.divider()
    st.caption("**Try these questions:**")
    samples = [
        "Can my landlord evict me without notice?",
        "What are my rights when arrested?",
        "How do I file a consumer complaint?",
        "My employer fired me unfairly. Help?",
        "Can I get bail if arrested?",
    ]
    for s in samples:
        if st.button(s, use_container_width=True, key=s):
            st.session_state["prefill"] = s

    st.divider()
    st.caption("Built by Harsha | BTech CSE")
    st.caption("1,625 Indian court judgements")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("⚖️ Nyaya Setu")
st.subheader("Free Legal Aid for Every Indian Citizen")
st.caption("Ask your legal question in English, Telugu, or Hindi.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Type your legal question here...") or prefill

if user_input:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.info("Get a free key at console.groq.com — takes 2 minutes.")
        st.stop()

    if not pipeline_loaded:
        st.error("Database not loaded. Please refresh the page.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching judgements..."):
            try:
                detected = detect_lang(user_input)
                query_en = user_input
                if detected != "en":
                    query_en = translate_text(user_input, target="en")

                chunks = retrieve(query_en, index, texts, metadatas, embed_model)
                answer = generate_answer(query_en, chunks, api_key, lang_code)

                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()

                st.markdown(answer)

                avg_score = np.mean([c["score"] for c in chunks])
                if avg_score >= 0.5:
                    st.markdown('<span class="confidence-high">⬤ Good match</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="confidence-low">⬤ Partial match — consult a lawyer</span>', unsafe_allow_html=True)

                st.caption(f"Based on {len(chunks)} Indian court judgements")

                with st.expander("📄 View source judgements"):
                    for c in chunks:
                        st.markdown(f"""<div class="source-card">
<strong>{c['metadata']['title']}</strong><br>
{c['metadata']['court']} | Relevance: {c['score']:.0%}<br>
<em>{c['text'][:200]}...</em>
</div>""", unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")