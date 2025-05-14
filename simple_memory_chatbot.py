import os
import json
import faiss
import numpy as np
import streamlit as st
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    chat_model = os.getenv("CHAT_MODEL", "llama3")
    temperature = 0.7
    max_tokens = 800
    memory_k = 5
    history_limit = 20
    vector_dim = 1536
    index_path = "data/faiss_index"
    user_identity = os.getenv("USER_IDENTITY", "")
    app_name = os.getenv("APP_NAME", "Memory Chatbot")
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

settings = Settings()
os.makedirs("data", exist_ok=True)

class SimpleEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * settings.vector_dim for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * settings.vector_dim

class OllamaChat:
    def __init__(self, model: str = None, temperature: float = None, max_tokens: int = None):
        self.model = model or settings.chat_model
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens

    def generate_completion(self, prompt: str, **kwargs) -> str:
        try:
            url = "http://localhost:11434/v1/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"]
        except Exception as e:
            raise Exception(f"Error calling Ollama: {str(e)}")

class MemoryVectorStore:
    def __init__(self, embedding_model: Optional[SimpleEmbedder] = None, index_path: str = None):
        self.embedding_model = embedding_model or SimpleEmbedder()
        self.index_path = index_path or settings.index_path
        self.metadata_path = f"{self.index_path}_metadata.json"
        self.vector_dim = settings.vector_dim
        self._initialize_index()

    def _initialize_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception:
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.metadata = []

    def save(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving index: {str(e)}")

    def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> int:
        embedding = self.embedding_model.embed_query(text)
        embedding_np = np.array([embedding]).astype(np.float32)
        self.index.add(embedding_np)
        memory_id = len(self.metadata)
        memory_metadata = {
            "id": memory_id,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        self.metadata.append(memory_metadata)
        self.save()
        return memory_id

    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        if not self.metadata:
            return []
        k = k or settings.memory_k
        k = min(k, len(self.metadata))
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding_np = np.array([query_embedding]).astype(np.float32)
        distances, indices = self.index.search(query_embedding_np, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                memory = self.metadata[idx].copy()
                memory["distance"] = float(distances[0][i])
                memory["similarity"] = 1.0 / (1.0 + float(distances[0][i]))
                results.append(memory)
        return results

class ConversationMemory:
    def __init__(self, history_limit: int = None):
        self.history_limit = history_limit or settings.history_limit
        self.history = []

    def add_user_message(self, message: str):
        return self._add_message("user", message)

    def add_assistant_message(self, message: str):
        return self._add_message("assistant", message)

    def _add_message(self, role: str, content: str):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(message)
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
        return message

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history.copy()

    def get_formatted_history(self) -> str:
        return "\n".join([f"{'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in self.history])

class MemoryChatbot:
    def __init__(self):
        self.memory = ConversationMemory()
        self.vector_store = MemoryVectorStore()
        self.model = OllamaChat()

    def process_input(self, user_input: str) -> Dict[str, Any]:
        self.memory.add_user_message(user_input)
        prompt = self._create_prompt(user_input)
        response = self.model.generate_completion(prompt)
        self.memory.add_assistant_message(response)
        self.vector_store.add_memory(f"Human: {user_input}\nAssistant: {response}")
        return {
            "response": response,
            "history": self.memory.get_history()
        }

    def _create_prompt(self, query: str) -> str:
        return f"""
You are a helpful assistant.

{self.memory.get_formatted_history()}

Human: {query}
Assistant:"""

# Streamlit App
st.set_page_config(page_title=settings.app_name, layout="wide")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = MemoryChatbot()

st.title(settings.app_name)
user_input = st.chat_input("Type your message here...")

if user_input:
    result = st.session_state.chatbot.process_input(user_input)
    for msg in result["history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Sidebar memory viewer
st.sidebar.header("🧠 Memory Control")
if st.sidebar.button("🔍 View Saved Memories"):
    memories = st.session_state.chatbot.vector_store.metadata
    for mem in memories:
        st.sidebar.markdown(f"**{mem['timestamp']}**\n\n{mem['text']}\n\n---")

if st.sidebar.button("🗑️ Clear All Memories"):
    st.session_state.chatbot.vector_store._create_new_index()
    st.session_state.chatbot.vector_store.save()
    st.sidebar.success("Memory cleared.")
