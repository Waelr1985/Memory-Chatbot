# 🧠 Memory Chatbot with Ollama

A privacy-first, locally hosted chatbot app that remembers your conversations and retrieves relevant context using vector memory. Powered by [Streamlit](https://streamlit.io/) and [Ollama](https://ollama.com/) for self-hosted LLM responses.

---

## 🚀 Features

### ✅ Local LLM Inference
- Uses **Ollama** to run models like `llama3` entirely on your machine.
- No internet or API keys required for language generation.

### 🧠 Long-Term Memory (FAISS)
- Saves interactions (human + assistant) as vector embeddings.
- Recalls relevant past exchanges to improve contextual response.
- View and manage memory from the sidebar.

### 💬 Chat Interface
- Clean, real-time chatbot UI powered by Streamlit.
- All conversation history is displayed in context.

### 🗂️ Persistent Storage
- Stores memory in `data/faiss_index` and metadata in JSON.
- Chat history persists across sessions.

---

## 📦 Requirements

- Python 3.8+
- FAISS
- Streamlit
- NumPy
- Requests
- [Ollama](https://ollama.com/) installed and running locally

---

## 🛠️ Installation & Run

1. **Clone the repository**
```bash
git clone https://github.com/waelr1985/memory-chatbot.git
cd memory-chatbot
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Ollama with your model (e.g., llama3)**
```bash
ollama run llama3
```

5. **Run the chatbot app**
```bash
streamlit run simple_memory_chatbot.py
```

---

## 📁 File Structure

```
├── data/                         # Contains FAISS index + metadata
├── simple_memory_chatbot.py     # Main app script
├── .env                         # Optional config (e.g., APP_NAME, USER_IDENTITY)
├── requirements.txt             # Python dependencies
```

---

## ✨ Customization

- Change model: edit `CHAT_MODEL` in `.env` (e.g., `llama3`, `mistral`) 
- Modify temperature/max_tokens in `Settings`
- Replace the embedder with real embeddings (e.g., `sentence-transformers`)

---

## 🔐 Privacy
- This app runs **entirely offline**.
- No data is sent to third-party APIs.
- You control your memory.

---

## 🤖 Future Ideas
- Support multiple users
- Add local embedding models
- Export/import memory
- Use LangChain for prompt chaining

---

## 📄 License
MIT License. Feel free to fork and enhance!

---

## 🙋‍♂️ Author
Wael Rahhal — [LinkedIn](https://www.linkedin.com/in/wael-rahhal-ph-d-06786522/) | [GitHub](https://github.com/waelr1985) 
