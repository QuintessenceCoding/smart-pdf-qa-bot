# 📄 Smart PDF Q&A Bot 🤖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-pdf-app-bot-3gwubr78brfpi8jufga6v4.streamlit.app/)

Upload a PDF (like notes, books, resumes) and **chat with it** using AI. Ask questions and get instant, voice-assisted answers — with references from your document!

---

## 🔍 Features

- 📤 Upload any PDF
- 💬 Ask questions in natural language
- 🧠 Google Gemini + embeddings for accurate answers
- 📚 Context-aware responses from top relevant text chunks
- 🔊 Text-to-speech (gTTS) for accessibility
- 📘 One-click PDF summarization

---

## 🚀 Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/smart-pdf-qa-bot.git
cd smart-pdf-qa-bot
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate (Windows)
pip install -r requirements.txt
streamlit run app.py
