import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

# --------------------------------
# 🏁 Streamlit App Configuration
# --------------------------------
st.set_page_config(page_title="SHA Chatbot", page_icon="💬", layout="centered")

# --------------------------------
# 🧾 Create FAQ data manually (no external CSV needed)
# --------------------------------
faq_data = [
    {"question": "What is the Social Health Authority?",
     "answer": "The Social Health Authority (SHA) is a government organization responsible for managing universal health coverage in Kenya."},
    {"question": "How can I register for the Social Health Authority?",
     "answer": "You can register online through the SHA portal or visit the nearest SHA office with your identification documents."},
    {"question": "Who is eligible to register for the Social Health Authority?",
     "answer": "All Kenyan citizens and legal residents are eligible to register for the Social Health Authority."},
    {"question": "What identification documents are required for registration?",
     "answer": "You will need your National ID for adults or a birth certificate for children to register for SHA."},
    {"question": "Can children be registered for the Social Health Authority?",
     "answer": "Yes, children can be registered under their parents' or guardians' SHA accounts."},
    {"question": "What are the benefits covered under the Social Health Authority?",
     "answer": "SHA covers outpatient, inpatient, maternity, emergency, and chronic disease management services."},
    {"question": "How are contributions to the Social Health Authority made?",
     "answer": "Members can make monthly contributions via M-Pesa Paybill or automatic payroll deductions."},
    {"question": "Are employers required to contribute to the Social Health Authority for their employees?",
     "answer": "Yes, employers are legally required to contribute to SHA on behalf of their employees."},
    {"question": "What healthcare services are purchased by the Social Health Authority?",
     "answer": "SHA purchases preventive, promotive, curative, rehabilitative, and palliative care services."},
    {"question": "How can beneficiaries access healthcare services under the fund?",
     "answer": "Beneficiaries can access care from empaneled and contracted healthcare providers using their SHA ID or biometric verification."},
    {"question": "Who qualifies as a dependent under the Social Health Authority?",
     "answer": "Dependents include spouses, children under 18, or up to 25 if in school, and disabled dependents regardless of age."},
    {"question": "Can indigent and vulnerable persons receive coverage under the Social Health Authority?",
     "answer": "Yes, indigent and vulnerable persons are covered through government-funded subsidies."},
    {"question": "What is the role of empaneled and contracted healthcare providers?",
     "answer": "They provide healthcare services to SHA members and ensure quality standards are maintained."},
    {"question": "How does the Social Health Authority ensure the quality of healthcare services?",
     "answer": "SHA regularly audits facilities, monitors outcomes, and ensures adherence to healthcare standards."},
    {"question": "What are the obligations of households in relation to the Social Health Authority?",
     "answer": "Households must register, keep information updated, and make timely contributions."},
    {"question": "Are there any specific benefits for chronic and critical illnesses?",
     "answer": "Yes, SHA provides special coverage for chronic and critical illnesses under specialized care packages."},
    {"question": "How can I list beneficiaries under my Social Health Authority coverage?",
     "answer": "You can add beneficiaries through the SHA online portal or by visiting an SHA office."},
    {"question": "What happens if a member fails to make contributions to the fund?",
     "answer": "Failure to pay contributions may lead to suspension of benefits until arrears are cleared."}
]

faq_df = pd.DataFrame(faq_data)

# --------------------------------
# ⚙️ Preprocess and Vectorize FAQs
# --------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(faq_df["question"])

# --------------------------------
# 💬 Chatbot Response Function
# --------------------------------
def chatbot_response(user_input):
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    index = similarities.argmax()
    score = similarities[0, index]

    if score < 0.3:
        return "I'm sorry, I couldn’t find a relevant answer. Please contact SHA support for more assistance."
    else:
        return faq_df.iloc[index]["answer"]

# --------------------------------
# 🗃️ Log Interaction Function
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
# 🎨 Streamlit UI
# --------------------------------
st.title("💬 Social Health Authority Chatbot")
st.markdown("This chatbot provides quick answers to **frequently asked questions about the Social Health Authority (SHA)** in Kenya.")

# Sidebar with logo
ASSETS_DIR = "assets"
logo_path = os.path.join(ASSETS_DIR, "sha_logo.png")

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
else:
    st.sidebar.markdown("### 🏥 SHA Chatbot")

st.sidebar.markdown("---")
st.sidebar.info("Ask me about SHA registration, eligibility, benefits, or contributions.")

# Chat interface
st.markdown("### 🧠 Ask Your Question")
user_query = st.text_input("Type your question below:")

if st.button("Get Answer"):
    if user_query.strip():
        response = chatbot_response(user_query)
        st.success(response)
        log_chat(user_query, response)  # Log conversation
    else:
        st.warning("Please enter a question before submitting.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Scikit-learn, and TF-IDF similarity.")
