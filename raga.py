import streamlit as st
import os
import json
from env import GEMINI_API_KEY
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, load_summarize_chain

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Gemini RAG-A", page_icon="🤖", layout="wide")
genai.configure(api_key=GEMINI_API_KEY)
st.title("📘 Gemini RAG-A — Explainable & Feedback-Aware Q&A")

# ---------------- INPUTS ----------------
query = st.text_input("🔎 Enter your question:")
models = [m.name for m in list(genai.list_models()) if "generateContent" in m.supported_generation_methods]
model_name = st.selectbox("Select Gemini Model", [m for m in models if "gemini" in m.lower()])

docs_folder = "docs"
os.makedirs(docs_folder, exist_ok=True)
doc_files = [f for f in os.listdir(docs_folder) if f.endswith((".pdf", ".docx"))]
selected_doc = st.selectbox("Select a document from docs folder", doc_files)
answer_btn = st.button("⚡ Get Answer")

# ---------------- FUNCTIONS ----------------
def load_feedback():
    if os.path.exists("feedback.json"):
        with open("feedback.json", "r") as f:
            return json.load(f)
    return {}

def save_feedback(data):
    with open("feedback.json", "w") as f:
        json.dump(data, f, indent=2)

def update_feedback(query, answer, sentiment):
    data = load_feedback()
    data[query] = {"answer": answer, "feedback": sentiment}
    save_feedback(data)

# ---------------- SESSION STATE INIT ----------------
if "feedback_clicked" not in st.session_state:
    st.session_state.feedback_clicked = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "vectordb_cache" not in st.session_state:
    st.session_state.vectordb_cache = {}

# Reset feedback state if query changes
if query != st.session_state.last_query:
    st.session_state.feedback_clicked = False
    st.session_state.last_query = query

# ---------------- MAIN LOGIC ----------------
if answer_btn and selected_doc and query:
    file_path = os.path.join(docs_folder, selected_doc)

    with st.spinner("🧠 Analyzing the document..."):
        # --- Load and split document ---
        if selected_doc.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(raw_docs)

        # --- Create or load embeddings (cached in session_state) ---
        persist_dir = f"chroma_db/{selected_doc}"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)

        if selected_doc not in st.session_state.vectordb_cache:
            if not os.path.exists(persist_dir):
                vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
                vectordb.persist()
            else:
                vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            st.session_state.vectordb_cache[selected_doc] = vectordb
        else:
            vectordb = st.session_state.vectordb_cache[selected_doc]

        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.2, "k": 5}
        )

        # --- RAG-A Prompt ---
        rag_a_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an explainable AI assistant answering based only on the provided document context.
Each section of context is labeled [Source i].
When giving your answer, cite relevant sources in square brackets (e.g., [Source 1], [Source 3]).
If you cannot find the answer, say clearly that it’s not available.

Context:
{context}

Question:
{question}

Answer (with citations):
            """
        )

        # --- Build RAG-A Chain ---
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY, temperature=0.3)
        combine_chain = load_summarize_chain(
            llm, 
            chain_type="stuff", 
            prompt=rag_a_prompt,
            document_variable_name="context"  # <--- FIX: matches prompt variable
        )
        qa = RetrievalQA(retriever=retriever, combine_documents_chain=combine_chain, return_source_documents=True)

        # --- Run Query ---
        result = qa({"query": query})
        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])

    # ---------------- DISPLAY RESULTS ----------------
    st.divider()
    if not answer:
        st.error("❌ No relevant answer found.")
    else:
        st.subheader("💡 Answer")
        st.markdown(answer)

        # --- Feedback buttons ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 This is helpful") and not st.session_state.feedback_clicked:
                update_feedback(query, answer, "positive")
                st.session_state.feedback_clicked = True
                st.success("✅ Feedback recorded! Thank you.")
        with col2:
            if st.button("👎 Not accurate") and not st.session_state.feedback_clicked:
                update_feedback(query, answer, "negative")
                st.session_state.feedback_clicked = True
                st.warning("⚠️ Feedback recorded — we'll use it to improve future answers.")

        # --- Source display with confidence ---
        if sources:
            st.subheader("📚 Retrieved Source Paragraphs")
            for i, doc in enumerate(sources, 1):
                score = getattr(doc, "score", None)
                st.markdown(f"**[Source {i}]** — {doc.metadata.get('source', selected_doc)}")
                if score:
                    st.caption(f"Confidence: {score:.2f}")
                with st.expander(f"View Source {i}"):
                    st.write(doc.page_content)
        else:
            st.info("No explicit source paragraphs found for this answer.")

# ---------------- FEEDBACK DASHBOARD ----------------
if st.sidebar.button("📊 View Feedback Summary"):
    data = load_feedback()
    if not data:
        st.sidebar.info("No feedback yet.")
    else:
        pos = sum(1 for f in data.values() if f["feedback"] == "positive")
        neg = sum(1 for f in data.values() if f["feedback"] == "negative")
        st.sidebar.metric("👍 Positive Feedback", pos)
        st.sidebar.metric("👎 Negative Feedback", neg)
        st.sidebar.json(data)
