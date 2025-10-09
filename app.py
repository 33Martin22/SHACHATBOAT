# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from datetime import datetime

# ------------------------
# Config
# ------------------------
MODELS_DIR = "models"
LOGS_DIR = "logs"
ASSETS_DIR = "assets"
os.makedirs(LOGS_DIR, exist_ok=True)

VECT_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
FAQ_CSV_PATH = os.path.join(MODELS_DIR, "faq_data_clean.csv")
ANALYTICS_CSV = os.path.join(LOGS_DIR, "analytics.csv")

CONFIDENCE_DEFAULT = 0.30  # default threshold

# ------------------------
# Helpers
# ------------------------
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def append_analytics(row: dict):
    header = ["timestamp","user_query","matched_intent","matched_question","matched_response","score","accepted"]
    write_header = not os.path.exists(ANALYTICS_CSV)
    df = pd.DataFrame([row])
    if write_header:
        df.to_csv(ANALYTICS_CSV, index=False, mode='w', columns=header)
    else:
        df.to_csv(ANALYTICS_CSV, index=False, mode='a', header=False, columns=header)

# ------------------------
# Load models and faq
# ------------------------
@st.cache_resource
def load_models():
    if not (os.path.exists(VECT_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(FAQ_CSV_PATH)):
        return None, None, None
    vectorizer = pickle.load(open(VECT_PATH, "rb"))
    tfidf_matrix = pickle.load(open(TFIDF_PATH, "rb"))
    faq_df = pd.read_csv(FAQ_CSV_PATH, dtype=str).fillna("")
    return vectorizer, tfidf_matrix, faq_df

vectorizer, tfidf_matrix, faq_df = load_models()

# ------------------------
# Page layout & styles
# ------------------------
st.set_page_config(page_title="SHA Chatbot", page_icon="assets/shalogo.png" if os.path.exists(os.path.join(ASSETS_DIR,"shalogo.png")) else "💬", layout="centered")
st.markdown("""
    <style>
    .chat-container { max-height: 520px; overflow-y: auto; padding: 12px; border-radius: 10px; margin-bottom: 12px; }
    .user-bubble { display:flex; justify-content:flex-end; margin:8px 0; }
    .bot-bubble  { display:flex; justify-content:flex-start; margin:8px 0; }
    .bubble { max-width:80%; padding:10px 14px; border-radius:12px; line-height:1.4; }
    .user { background-color:#0d6efd; color: white; border-radius: 12px 12px 0 12px; }
    .bot  { background-color:#e9ecef; color: #000; border-radius: 12px 12px 12px 0; }
    .meta { font-size:12px; color: #666; margin-top:6px; }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.image(os.path.join(ASSETS_DIR,"sha_logo.png") if os.path.exists(os.path.join(ASSETS_DIR,"sha_logo.png")) else None, width=150)
st.sidebar.title("SHA Chatbot — Settings")
threshold = st.sidebar.slider("Match confidence threshold", 0.0, 1.0, value=float(CONFIDENCE_DEFAULT), step=0.01)
show_analytics = st.sidebar.checkbox("Show analytics file link", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("To update FAQs: edit `data/sha_faq.csv` and run `python train_model.py` to rebuild models.")

# ------------------------
# Header
# ------------------------
col1, col2, col3 = st.columns([1,6,1])
with col1:
    if os.path.exists(os.path.join(ASSETS_DIR,"sha_logo.png")):
        st.image(os.path.join(ASSETS_DIR,"sha_logo.png"), width=72)
with col2:
    st.markdown("<h2 style='margin:0; text-align:center'>💬 SHA FAQ Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color: #6c757d;'>Ask questions about Social Health Authority policies, registration, benefits and more.</div>", unsafe_allow_html=True)
with col3:
    if os.path.exists(os.path.join(ASSETS_DIR,"sha_logo.png")):
        st.write("")

# ------------------------
# Model check
# ------------------------
if vectorizer is None:
    st.error("Model artifacts not found. Please run `python train_model.py` first to generate the models.\n\nThen refresh this app.")
    st.stop()

# ------------------------
# Chat session state
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of tuples (sender,msg)
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# ------------------------
# Input
# ------------------------
user_input = st.chat_input("Type your question about SHA (e.g. 'How can I register?')")

if user_input:
    cleaned = preprocess(user_input)
    query_vec = vectorizer.transform([cleaned])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]  # shape (n_faqs,)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    matched_question = faq_df.iloc[best_idx]['question']
    matched_response = faq_df.iloc[best_idx]['response']
    matched_intent = faq_df.iloc[best_idx].get('intent', "")

    # logging analytics row
    analytics_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_query": user_input,
        "matched_intent": matched_intent,
        "matched_question": matched_question,
        "matched_response": matched_response,
        "score": best_score,
        "accepted": bool(best_score >= threshold)
    }
    append_analytics(analytics_row)

    # show "typing" spinner while creating response
    with st.spinner("SHA is typing..."):
        time.sleep(0.6)  # small delay to simulate typing (adjust as needed)

    if best_score >= threshold:
        response = matched_response
    else:
        response = "Sorry — I don't have a confident answer for that. Please contact SHA support or try rephrasing your question."

    # update history
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", response))
    st.session_state.last_query = user_input

# ------------------------
# Display chat history (scrollable)
# ------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, msg in st.session_state.history:
    if sender == "user":
        st.markdown(f"""
            <div class="user-bubble">
                <div class="bubble user">{msg}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="bot-bubble">
                <div class="bubble bot">{msg}</div>
            </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------
# Show analytics link / download
# ------------------------
if show_analytics:
    if os.path.exists(ANALYTICS_CSV):
        with open(ANALYTICS_CSV, "rb") as f:
            data = f.read()
        st.download_button(label="Download analytics CSV", data=data, file_name="analytics.csv", mime="text/csv")
    else:
        st.info("No analytics data yet. Interact with the bot to generate analytics logs.")
