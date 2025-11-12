import streamlit as st
import random
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="SHA Chatbot", page_icon="💬", layout="wide")

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f8f9fa, #e3f2fd);
    padding: 2rem;
    border-radius: 12px;
}
.chat-container {
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 1.5rem;
    min-height: 500px;
}
.user-msg {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
    width: fit-content;
    max-width: 80%;
}
.bot-msg {
    background-color: #E9E9EB;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
    width: fit-content;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.markdown("## 💬 **SHA Chatbot**")
    st.markdown("""
    Welcome to your smart assistant!  
    ---
    - Uses **TF-IDF + Cosine Similarity** for matching  
    - Understands **synonyms & related words**  
    - Clean and image-free UI  
    ---
    """)

# ------------------- SYNONYMS -------------------
synonyms = {
    "register": ["enroll", "signup", "join", "apply"],
    "payment": ["contribution", "premium", "fee"],
    "benefit": ["advantage", "support", "coverage"],
    "member": ["user", "participant", "beneficiary"],
    "claim": ["compensation", "reimbursement", "refund"],
    "hospital": ["clinic", "facility", "health center"],
    "dependents": ["family", "spouse", "children"],
    "update": ["edit", "change", "modify"],
    "card": ["membership card", "id card"],
}

def expand_synonyms(text):
    for key, syns in synonyms.items():
        for s in syns:
            text = re.sub(rf"\b{s}\b", key, text)
    return text

# ------------------- KNOWLEDGE BASE -------------------
intents = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening"
        ],
        "responses": [
            "Hello! 👋 How can I assist you today?",
            "Hi there! 😊 What would you like to know?",
            "Hey! Ready to learn more about SHA?"
        ]
    },
    "registration": {
        "patterns": [
            "how do i register", "how can i join sha", "membership registration", 
            "how to enroll", "signup process", "apply for membership"
        ],
        "responses": [
            "To register with SHA, visit our online portal or the nearest SHA office. You'll need your national ID, phone number, and a recent passport photo. Once submitted, you’ll receive your SHA membership number within 48 hours. 🧾",
            "Registration is simple! Head to our official website and click **‘Register Now’**. Fill out your personal details, upload a valid ID, and pay your initial contribution. Your account will be activated once verified. ✅"
        ]
    },
    "payments": {
        "patterns": [
            "how do i make a payment", "how can i pay", "monthly contribution", 
            "premium payment", "sha fees", "payment methods"
        ],
        "responses": [
            "You can make payments via M-Pesa using Paybill **XXXXXX**, account number as your SHA ID. 💸",
            "Payments can be made monthly, quarterly, or annually through mobile money or bank deposit. Check your member dashboard for your current balance. 💰"
        ]
    },
    "benefits": {
        "patterns": [
            "what are the benefits", "sha advantages", "coverage details", 
            "what do i get", "benefit packages"
        ],
        "responses": [
            "SHA members enjoy coverage for outpatient, inpatient, maternity, dental, and optical services. Additional benefits include annual health checkups and mental health support. ❤️",
            "Our benefits include affordable healthcare access, preventive screenings, and dependents coverage. Members also get emergency and chronic care support."
        ]
    },
    "dependents": {
        "patterns": [
            "how do i add dependents", "register my family", "add my spouse", 
            "include children", "dependents registration"
        ],
        "responses": [
            "You can add dependents by logging into your SHA account and selecting **‘Manage Dependents’**. Provide their details and upload supporting documents (birth or marriage certificate). 👨‍👩‍👧‍👦",
            "Dependents can be added at any time through your member portal or by visiting the nearest branch."
        ]
    },
    "claims": {
        "patterns": [
            "how do i make a claim", "claim process", "reimbursement procedure", 
            "how to get refund", "medical claim"
        ],
        "responses": [
            "To make a claim, log into your portal and select **‘Submit Claim’**. Upload your hospital receipts and medical reports. Claims are reviewed within 5–10 business days. 💼",
            "Claims can also be submitted manually at any SHA branch. Ensure you include your member ID, treatment summary, and payment proof."
        ]
    },
    "card_replacement": {
        "patterns": [
            "lost my card", "replace membership card", "how to get new card", 
            "missing sha id", "card renewal"
        ],
        "responses": [
            "If your card is lost, visit the nearest SHA office with your ID for verification. A replacement card costs KES 200 and will be issued within 3 days. 🪪",
            "You can request a new card online by selecting **‘Replace Card’** in your member dashboard."
        ]
    },
    "contact": {
        "patterns": [
            "how can i contact you", "support email", "customer service", 
            "help desk", "office location"
        ],
        "responses": [
            "You can reach us via email at **support@sha.ai** or call our helpline **+254 700 000 000**. 📞",
            "Visit our main office at SHA House, Nairobi, open Monday–Friday 8am–5pm."
        ]
    },
    "default": {
        "patterns": [],
        "responses": [
            "I’m not sure I understood that 🤔. Could you rephrase your question?",
            "Hmm, I don’t have info on that yet. Try asking about registration, benefits, or payments."
        ]
    }
}

# ------------------- INTENT MATCHING -------------------
def find_best_intent(user_input):
    user_input = expand_synonyms(user_input.lower())

    patterns = []
    labels = []
    for intent, data in intents.items():
        for p in data["patterns"]:
            patterns.append(expand_synonyms(p.lower()))
            labels.append(intent)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(patterns)
    user_vec = vectorizer.transform([user_input])
    sims = cosine_similarity(user_vec, tfidf_matrix)
    best_idx = np.argmax(sims)
    best_intent = labels[best_idx]
    best_score = sims[0][best_idx]

    if best_score < 0.35:
        return "default"
    return best_intent

# ------------------- CHAT INTERFACE -------------------
st.markdown("<div class='main'><div class='chat-container'>", unsafe_allow_html=True)

user_input = st.text_input("💬 Type your message:", key="user_input")

if user_input:
    intent = find_best_intent(user_input)
    response = random.choice(intents[intent]["responses"])
    st.markdown(f"<div class='user-msg'>🧑‍💻 {user_input}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-msg'>🤖 {response}</div>", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)
