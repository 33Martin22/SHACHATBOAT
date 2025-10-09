import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import time
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# File Paths
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ------------------------
# Text Preprocessing
# ------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------
# Load Models and Data
# ------------------------
@st.cache_resource
def load_models():
    faq_df = pd.read_csv(os.path.join(MODELS_DIR, "faq_data_clean.csv"))
    vectorizer = pickle.load(open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb"))
    tfidf_matrix = pickle.load(open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "rb"))
    return faq_df, vectorizer, tfidf_matrix

faq_df, vectorizer, tfidf_matrix = load_models()

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(page_title="SHA Chatbot", page_icon="💬", layout="centered")

# ------------------------
# Sidebar (Theme + Logo)
# ------------------------
logo_path = os.path.join(ASSETS_DIR, "sha_logo.png")

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
else:
    st.sidebar.warning("⚠️ Logo not found. Please add 'sha_logo.png' in the assets folder.")

st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
st.sidebar.markdown("---")
st.sidebar.markdown("**About:** This chatbot answers FAQs about the **Social Health Authority (SHA)** in Kenya.")

# ------------------------
# Custom CSS (GPT-like)
# ------------------------
bg_color = "#f7f7f8" if theme == "Light" else "#1e1e1e"
user_color = "#0056b3" if theme == "Light" else "#007bff"
bot_color = "#e5e5ea" if theme == "Light" else "#2c2c2c"
text_color = "black" if theme == "Light" else "white"

st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .chat-container {{
        max-height: 500px;
        overflow-y: auto;
        padding: 20px;
        border-radius: 12px;
        background-color: {bg_color};
        width: 90%;
        margin: auto;
    }}
    .user-bubble {{
        background-color: {user_color};
        color: white;
        border-radius: 15px 15px 0 15px;
        padding: 10px 15px;
        margin: 10px 0;
        text-align: right;
    }}
    .bot-bubble {{
        background-color: {bot_color};
        color: {text_color};
        border-radius: 15px 15px 15px 0;
        padding: 10px 15px;
        margin: 10px 0;
        text-align: left;
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Header Section
# ------------------------
if os.path.exists(logo_path):
    st.image(logo_path, width=100)
st.markdown(f"<h2 style='text-align:center; color:{text_color}'>💬 SHA Chatbot</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:{text_color}'>Your intelligent assistant for Social Health Authority FAQs.</p>", unsafe_allow_html=True)

# ------------------------
# Chat Logic
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me anything about SHA...")

if user_input:
    clean_query = preprocess(user_input)
    query_vec = vectorizer.transform([clean_query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    best_idx = np.argmax(similarity)
    best_score = similarity[0, best_idx]

    if best_score < 0.3:
        bot_response = "I'm sorry, I don't have that answer yet. Please contact SHA support for more details."
    else:
        bot_response = faq_df.iloc[best_idx]['response']

    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", bot_response))

# ------------------------
# Display Chat
# ------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, message in st.session_state.history:
    if sender == "user":
        st.markdown(f"<div class='user-bubble'>🧑‍💻 {message}</div>", unsafe_allow_html=True)
    else:
        with st.spinner("Typing..."):
            time.sleep(0.7)
        st.markdown(f"<div class='bot-bubble'>🤖 {message}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
