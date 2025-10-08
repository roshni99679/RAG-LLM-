import streamlit as st
import os
from env import GEMINI_API_KEY
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- Configure Gemini API ---
genai.configure(api_key=GEMINI_API_KEY)

# --- Streamlit App ---
st.set_page_config(page_title="Gemini RAG App", page_icon="🤖", layout="wide")
st.title("📘 RAG with Gemini — Explainable Q&A")

# --- User Inputs ---
query = st.text_input("🔎 Enter your question:")
models = [m.name for m in list(genai.list_models()) if "generateContent" in m.supported_generation_methods]
model_name = st.selectbox("Select Gemini Model", [m for m in models if "gemini" in m.lower()])

docs_folder = "docs"
os.makedirs(docs_folder, exist_ok=True)
doc_files = [f for f in os.listdir(docs_folder) if f.endswith((".pdf", ".docx"))]
selected_doc = st.selectbox("Select a document from docs folder", doc_files)

answer_btn = st.button("⚡ Get Answer")

# --- Main logic ---
if answer_btn and selected_doc and query:
    file_path = os.path.join(docs_folder, selected_doc)

    with st.spinner("🧠 Reading and analyzing the document..."):
        # --- Load document ---
        if selected_doc.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()

        # --- Split into manageable chunks ---
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(raw_docs)

        # --- Create or load embeddings persistently ---
        persist_directory = f"chroma_db/{selected_doc}"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)

        if not os.path.exists(persist_directory):
            vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
            vectordb.persist()
        else:
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # --- Retriever with relaxed threshold ---
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 5}
        )

        # --- Debug: See what was retrieved ---
        st.markdown("### 🔍 Retrieved Chunks (for debugging)")
        retrieved_docs = retriever.get_relevant_documents(query)
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                st.markdown(f"**{i}.** {doc.page_content[:500]}...")
                st.caption(f"Source: {doc.metadata.get('source', selected_doc)}")
        else:
            st.warning("⚠️ No relevant chunks retrieved. Try rephrasing your question or lowering the score threshold.")

        # --- LLM setup ---
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY, temperature=0.3)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # --- Get answer ---
        result = qa({"query": query})
        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])

    # --- Display results ---
    st.divider()
    if not answer:
        st.error("❌ No relevant answer found.")
    else:
        st.subheader("💡 Answer")
        st.markdown(answer)

        if sources:
            st.subheader("📚 Source Paragraphs")
            for i, doc in enumerate(sources, 1):
                st.markdown(f"> {doc.page_content}")
                st.caption(f"Source: {doc.metadata.get('source', selected_doc)}")
        else:
            st.info("No explicit source paragraphs found for this answer.")
