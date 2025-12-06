import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import google.generativeai as genai
import os
import time

# --------------------------------
# 🌐 Page Configuration
# --------------------------------
st.set_page_config(page_title="SHA Chatbot", page_icon="💬", layout="centered")

# --------------------------------
# 🔑 Configure Gemini API
# --------------------------------
GEMINI_API_KEY = "AIzaSyBy2ToDOl5BAfyRLLQNVXQvbSf0SY-mSEw"
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# --------------------------------
# 🧾 FAQ Dataset
# --------------------------------
faq_data = [
    {"question": "What is the Social Health Authority?",
     "answer": "The Social Health Authority (SHA) is a government agency in Kenya mandated to implement and manage Universal Health Coverage (UHC)."},
    {"question": "How can I register for the Social Health Authority?",
     "answer": "You can register through the official SHA portal or at any SHA office using your National ID."},
    {"question": "Who is eligible to register for SHA?",
     "answer": "All Kenyan citizens and legal residents, including informal workers and foreign residents with valid permits."},
    # (... your full FAQ dataset here ...)
]

faq_df = pd.DataFrame(faq_data)

# --------------------------------
# ⚙ TF-IDF Model
# --------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(faq_df["question"])

def get_gemini_answer(user_input):
    """Send unknown questions to Gemini Pro."""
    prompt = f"""
    You are an SHA (Social Health Authority Kenya) assistant.
    Answer the following user question accurately, clearly, and professionally:

    Question: {user_input}
    """

    response = gemini_model.generate_content(prompt)
    return response.text

def chatbot_response(user_input):
    """Returns best FAQ match OR uses Gemini API."""
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    index = similarities.argmax()
    score = similarities[0, index]

    # If high similarity → use preset answer
    if score >= 0.30:
        return faq_df.iloc[index]["answer"]

    # Otherwise → use Gemini API
    gemini_reply = get_gemini_answer(user_input)
    return gemini_reply

# --------------------------------
# 🗃 Chat Logging
# --------------------------------
def log_chat(user_input, bot_response):
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": user_input,
        "bot_response": bot_response
    }])
    if os.path.exists("chat_logs.csv"):
        log_entry.to_csv("chat_logs.csv", mode="a", index=False, header=False)
    else:
        log_entry.to_csv("chat_logs.csv", mode="w", index=False, header=True)

# --------------------------------
# 💬 Streamlit UI
# --------------------------------
st.markdown("""
    <h1 style='text-align: center;'>💬 Social Health Authority Chatbot</h1>
    <p style='text-align: center; color: gray;'>Your virtual assistant for SHA information and support in Kenya.</p>
    <hr>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🏥 About SHA Chatbot")
st.sidebar.info("Ask about registration, eligibility, benefits, contributions, or healthcare services.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])

# User input box
user_query = st.chat_input("Ask me something about SHA...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate bot response with typing effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = chatbot_response(user_query)
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.03)
        message_placeholder.markdown(full_response)

    # Save to session + log
    st.session_state.chat_history.append({"user": user_query, "bot": response})
    log_chat(user_query, response)
