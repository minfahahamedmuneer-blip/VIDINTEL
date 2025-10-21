# VidIntel - YouTube RAG Chatbot powered by Gemini + FAISS + SentenceTransformers

import os
import streamlit as st
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import yt_dlp
import concurrent.futures

# 🚀 Fix PyTorch file watch warning
os.environ["STREAMLIT_WATCHER_NON_POLLING"] = "true"

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "previous_video_url" not in st.session_state:
    st.session_state.previous_video_url = None

# Metadata extraction
def get_video_metadata(video_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'format': 'bestaudio/best'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
            return {
                "title": info.get("title", "Unknown"),
                "author": info.get("uploader", "Unknown"),
                "thumbnail": info.get("thumbnail", None),
                "length": info.get("duration", 0),
            }
        except Exception as e:
            st.error(f"⚠️ Failed to fetch metadata: {e}")
            return {"title": "Unknown", "author": "Unknown", "thumbnail": None, "length": 0}



# Captions

def get_video_captions(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception:
        return None

# Chunking

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    return splitter.split_text(text)

# Embedding + FAISS

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Search

def search_relevant_chunks(query, index, chunks):
    query_embedding = embedding_model.encode([query])
    _, closest_indices = index.search(query_embedding, k=5)
    return " ".join([chunks[i] for i in closest_indices[0]])

# Gemini response

def generate_response(query, relevant_text):
    prompt = f"User Query: {query}\n\nContext: {relevant_text}\n\nAnswer:"
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text if response else "Sorry, I couldn't generate a response."

# Reset state

def clear_chat():
    st.session_state.messages = []
    st.session_state.previous_video_url = None
    st.session_state.faiss_index = None
    st.session_state.embeddings = None

# UI
st.set_page_config(page_title="VidIntel", layout="wide")

with st.sidebar:
    st.title("Settings")
    dark_mode = st.toggle("Dark Mode", False)
    if st.button("Download Chat History"):
        chat_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button("Download", data=chat_text, file_name="chat_history.txt", mime="text/plain")
    st.markdown("---")
    st.info("VidIntel extracts and analyzes YouTube videos using AI.")

if dark_mode:
    st.markdown("""
    <style>
        .stApp { background-color: #222; color: white; }
        .stSidebar { background-color: #333; }
        .stButton > button { background-color: #444; color: white; border: 1px solid #666; }
        .stButton > button:hover { background-color: #555; }
        .stAlert { background-color: #666666; color: white; border-left: 5px solid #2c2c2c; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #505050; }
        ::-webkit-scrollbar-thumb { background: #505050; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #666666; }
    </style>
    """, unsafe_allow_html=True)

st.title("VIDINTEL 🤖")
video_url = st.text_input("Enter YouTube URL:")

if video_url:
    if st.session_state.previous_video_url != video_url:
        clear_chat()
        st.session_state.previous_video_url = video_url

    video_id = video_url.split("v=")[-1].split("&")[0]
    metadata = get_video_metadata(video_url)
    st.subheader(f"📌 Video: {metadata['title']} by {metadata['author']}")

    if metadata["thumbnail"]:
        st.image(metadata["thumbnail"], width=500)

    with st.spinner("🔍 Extracting transcript..."):
        captions = get_video_captions(video_id)

    if not captions:
        st.warning("\u26a0\ufe0f No captions found for this video.")
    else:
        chunks = chunk_text(captions)

        if not st.session_state.faiss_index:
            index, embeddings = embed_chunks(chunks)
            st.session_state.faiss_index = index
            st.session_state.embeddings = embeddings
        else:
            index = st.session_state.faiss_index

        if index is None:
            st.error("❌ FAISS index is not initialized properly.")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            query = st.chat_input("Ask something about the video...")

            if query:
                with st.chat_message("user"):
                    st.markdown(query)
                st.session_state.messages.append({"role": "user", "content": query})

                with st.spinner("🔎 Searching relevant information..."):
                    relevant_text = search_relevant_chunks(query, index, chunks)

                with st.spinner("💡 Generating response..."):
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(generate_response, query, relevant_text)
                        response = future.result()

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
