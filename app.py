import streamlit as st
import os
import numpy as np
import pandas as pd
from datetime import datetime
from env import GEMINI_API_KEY
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity

# --- Configure Gemini API ---
genai.configure(api_key=GEMINI_API_KEY)

# --- Streamlit setup ---
st.set_page_config(page_title="Gemini RAG App", page_icon="🤖", layout="wide")
st.title("📘 Adaptive RAG with Gemini — Explainable Q&A + Feedback Learning")

# --- User Inputs ---
query = st.text_input("🔎 Enter your question:")
models = [m.name for m in list(genai.list_models()) if "generateContent" in m.supported_generation_methods]
model_name = st.selectbox("Select Gemini Model", [m for m in models if "gemini" in m.lower()])

docs_folder = "docs"
os.makedirs(docs_folder, exist_ok=True)
doc_files = [f for f in os.listdir(docs_folder) if f.endswith((".pdf", ".docx"))]
selected_doc = st.selectbox("Select a document from docs folder", doc_files)

answer_btn = st.button("⚡ Get Answer")

# --- Feedback file setup ---
feedback_file = "feedback_log.csv"
if not os.path.exists(feedback_file):
    pd.DataFrame(columns=["timestamp", "query", "answer", "confidence", "feedback"]).to_csv(feedback_file, index=False)
else:
    feedback_data = pd.read_csv(feedback_file)
    if feedback_data.empty:
        feedback_data = pd.DataFrame(columns=["timestamp", "query", "answer", "confidence", "feedback"])

# --- Function to calculate feedback-based bias ---
def feedback_bias(query_embedding, feedback_df, embedding_model):
    """
    Adjusts retrieval weighting using past feedback.
    Positive feedback boosts similarity; negative feedback penalizes it.
    """
    if feedback_df.empty:
        return 0.0

    biases = []
    for _, row in feedback_df.iterrows():
        try:
            past_embedding = embedding_model.embed_query(row["query"])
            sim = cosine_similarity([query_embedding], [past_embedding])[0][0]
            # Boost if positive, penalize if negative
            weight = 1 if row["feedback"] == "positive" else -1
            biases.append(sim * weight)
        except Exception:
            continue

    if not biases:
        return 0.0
    return np.mean(biases)

# --- Main logic: Answering the query ---
if answer_btn and selected_doc and query:
    file_path = os.path.join(docs_folder, selected_doc)

    with st.spinner("🧠 Reading and analyzing the document..."):
        # --- Load document ---
        if selected_doc.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()

        # --- Split into chunks ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(raw_docs)

        # --- Embeddings setup ---
        persist_directory = f"chroma_db/{selected_doc}"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)

        if not os.path.exists(persist_directory):
            vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
            vectordb.persist()
        else:
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # --- Query embedding ---
        query_embedding = embeddings.embed_query(query)

        # --- Calculate feedback bias ---
        bias = feedback_bias(query_embedding, feedback_data, embeddings)
        st.caption(f"🧩 Feedback bias adjustment: {round(bias, 3)}")

        # --- Retrieve chunks with adjusted scoring ---
        retrieved_with_scores = vectordb.similarity_search_with_score(query, k=5)
        adjusted_results = []
        for doc, score in retrieved_with_scores:
            similarity = 1 / (1 + score)
            adjusted_confidence = max(0.0, min(1.0, similarity + bias))
            adjusted_results.append((doc, adjusted_confidence))

        adjusted_results.sort(key=lambda x: x[1], reverse=True)
        avg_conf = round(float(np.mean([conf for _, conf in adjusted_results])) if adjusted_results else 0.0, 2)

        # --- Store key info in session state ---
        st.session_state["last_query"] = query
        st.session_state["last_answer"] = None  # Placeholder until LLM answer
        st.session_state["last_conf"] = avg_conf
        st.session_state["last_sources"] = []

        if not adjusted_results:
            st.warning("⚠️ No relevant chunks retrieved. Try rephrasing your question.")
        else:
            st.subheader(f"🎯 Adjusted Confidence: **{avg_conf}**")
            st.progress(avg_conf)
            st.markdown("### 📄 Retrieved Chunks (with Feedback Learning)")
            for i, (doc, conf) in enumerate(adjusted_results, 1):
                st.markdown(f"**{i}. Confidence:** {conf}")
                st.markdown(f"> {doc.page_content[:500]}...")
                st.caption(f"Source: {doc.metadata.get('source', selected_doc)}")

        # --- LLM setup ---
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY, temperature=0.3)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

        # --- Get final answer ---
        result = qa({"query": query})
        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])

        # --- Update session state with answer ---
        st.session_state["last_answer"] = answer
        st.session_state["last_sources"] = sources

# --- Display final answer if available ---
if "last_answer" in st.session_state and st.session_state["last_answer"]:
    st.divider()
    st.subheader("💡 Answer")
    st.markdown(st.session_state["last_answer"])

    if st.session_state["last_sources"]:
        st.subheader("📚 Source Paragraphs")
        for i, doc in enumerate(st.session_state["last_sources"], 1):
            st.markdown(f"> {doc.page_content}")
            st.caption(f"Source: {doc.metadata.get('source', selected_doc)}")
    else:
        st.info("No explicit source paragraphs found for this answer.")

# --- Feedback Buttons ---
st.markdown("---")
st.subheader("🗳️ Was this answer helpful?")
col1, col2 = st.columns(2)

def save_feedback(feedback_type):
    if "last_query" in st.session_state and st.session_state["last_answer"]:
        new_feedback = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "query": st.session_state["last_query"],
            "answer": st.session_state["last_answer"],
            "confidence": st.session_state["last_conf"],
            "feedback": feedback_type
        }])
        new_feedback.to_csv(feedback_file, mode="a", header=False, index=False)
        st.success("✅ Feedback recorded!")
    else:
        st.error("❌ No answer available to give feedback on.")

with col1:
    if st.button("👍 Yes, it was helpful"):
        save_feedback("positive")

with col2:
    if st.button("👎 No, it was incorrect"):
        save_feedback("negative")
