#app.py
import os
import json
import streamlit as st
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from utils.pdf_reader import extract_text_from_pdf
from utils.text_splitter import split_text
from embedder import embed_text_chunks
from gtts import gTTS
import tempfile
import base64

# ✅ Streamlit setup
st.set_page_config(page_title="Smart PDF Q&A Bot", layout="centered")

# ✅ Load environment and models
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ App UI
st.title("📄 Smart PDF Q&A Bot")
st.markdown("Upload a PDF and ask anything about it!")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save temporarily
    file_path = "data/temp_uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF Uploaded!")

    # Extract text
    text = extract_text_from_pdf(file_path)
    chunks = split_text(text)
    embeddings = embed_text_chunks(chunks)
    embedding_tensor = torch.tensor(embeddings)

    question = st.text_input("Ask a question about the PDF:", placeholder="e.g. What is this PDF about?")

    if st.button("Get Answer") and question.strip():
        with st.spinner("Thinking..."):

            # Embed question and calculate similarity
            question_embedding = embedder.encode(question, convert_to_tensor=True)
            scores = util.cos_sim(question_embedding, embedding_tensor)[0]

            # Use top-K most relevant chunks, but don't exceed available chunks
            top_k = min(3, len(chunks))  # Avoid out-of-range errors
            top_indices = torch.topk(scores, top_k).indices.tolist()

            top_chunks = [chunks[i] for i in top_indices]
            context = "\n\n---\n\n".join(top_chunks)

            # Prompt Gemini
            prompt = f"""You're a helpful assistant. Use only the following context to answer the user's question.

Context:
{context}

Question:
{question}

Answer in a helpful and concise way."""

            response = model.generate_content(prompt)

            st.subheader("🧠 Answer:")
            st.write(response.text.strip())

            # Text-to-speech
            tts = gTTS(text=response.text.strip(), lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio_file = fp.name

            # Play audio in Streamlit
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')  

    # Bonus: Summarize PDF
    if st.button("Summarize PDF"):
        with st.spinner("Summarizing..."):
            summary_context = "\n\n".join(chunks[:5])
            summary_prompt = f"Please summarize the following content:\n\n{summary_context}"
            summary_response = model.generate_content(summary_prompt)

            st.subheader("📘 PDF Summary:")
            st.write(summary_response.text.strip())
            # Text-to-speech for summary
            tts = gTTS(text=summary_response.text.strip(), lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio_file = fp.name

            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')

    # Debug toggle
    #if st.checkbox("🔍 Show Debug Info"):
      #  st.subheader("📊 Debug Info")
      #  st.write("First Chunk Sample:", chunks[0][:500])
       # st.write("Embedding count:", len(embeddings))
       # st.write("First Question Similarity Score:", scores[top_indices[0]].item())

else:
    st.info("👆 Please upload a PDF to begin.")
