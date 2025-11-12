import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import time

# --------------------------------
# 🌐 Page Configuration
# --------------------------------
st.set_page_config(page_title="SHA Chatbot", page_icon="💬", layout="centered")

# --------------------------------
# 🧾 Expanded and Detailed FAQ Dataset
# --------------------------------
faq_data = [
    {"question": "What is the Social Health Authority?",
     "answer": "The Social Health Authority (SHA) is a government agency in Kenya mandated to implement and manage Universal Health Coverage (UHC). It ensures that all Kenyans have access to quality and affordable healthcare services without facing financial hardship. The Authority coordinates health insurance schemes, registers members, and contracts healthcare providers to deliver services."},
    
    {"question": "What is the main goal of the Social Health Authority?",
     "answer": "The main goal of SHA is to guarantee equitable access to comprehensive healthcare for all Kenyan residents. It aims to achieve financial protection by spreading health risks across the population and reducing out-of-pocket medical expenses. In essence, it ensures every Kenyan can seek healthcare when needed, regardless of their income level."},
    
    {"question": "How can I register for the Social Health Authority?",
     "answer": "You can register for SHA online through the official SHA portal or by visiting the nearest SHA office. Registration requires your National ID or birth certificate for minors. Once registered, you’ll receive a unique SHA membership number or digital ID, which you can use to access healthcare services across empaneled facilities."},
    
    {"question": "Who is eligible to register for SHA?",
     "answer": "All Kenyan citizens and legally recognized residents are eligible to register for SHA. This includes both employed and self-employed individuals, as well as informal sector workers. Foreign residents with valid permits may also join to access healthcare services under SHA’s framework."},
    
    {"question": "What documents do I need to register for SHA?",
     "answer": "You’ll need a valid National ID card for adults or a birth certificate for children. For foreign residents, a valid work or residence permit is required. You may also be asked to provide proof of residence and contact details to complete the registration."},
    
    {"question": "Can children be registered for SHA?",
     "answer": "Yes, children can be registered under their parents’ or guardians’ accounts. Parents simply need to add the child’s details through the SHA portal or at an SHA office. Children are covered for outpatient, inpatient, and preventive healthcare services, including immunization and basic treatment."},
    
    {"question": "What healthcare services are covered under SHA?",
     "answer": "SHA covers a wide range of healthcare services, including outpatient visits, inpatient treatment, maternity care, emergency services, chronic disease management, and rehabilitative care. Preventive and promotive services like immunization, family planning, and screening are also included to encourage early detection and wellness."},
    
    {"question": "How are contributions to SHA made?",
     "answer": "Contributions are made monthly either through M-Pesa Paybill, payroll deductions, or direct deposits. Employed individuals have their contributions automatically remitted by employers, while informal workers can pay directly via mobile money. The amount may vary based on income bracket and category of employment."},
    
    {"question": "Are employers required to contribute to SHA?",
     "answer": "Yes, all employers in Kenya are legally required to remit monthly contributions to SHA on behalf of their employees. The contribution is typically shared between the employer and employee. Employers must also ensure that new hires are registered promptly to maintain continuous health coverage."},
    
    {"question": "Can I register for SHA if I am self-employed?",
     "answer": "Yes, self-employed and informal sector workers are encouraged to register. They can make contributions directly via M-Pesa or at any SHA office. SHA provides flexible payment options to ensure affordability and continuity of healthcare access for informal workers."},
    
    {"question": "How do I add or remove dependents under SHA?",
     "answer": "Dependents can be added or removed through the SHA online portal or by visiting an SHA office. Eligible dependents include your spouse, children under 18 years (or up to 25 if in school), and persons with disabilities under your care. Each dependent must have their details properly registered in the system."},
    
    {"question": "Who qualifies as a dependent under SHA?",
     "answer": "Dependents include a legal spouse, biological or legally adopted children under 18 years, and children under 25 years if enrolled in school. Persons with disabilities who rely on the member for support also qualify regardless of age. These dependents enjoy similar benefits as the principal member."},
    
    {"question": "What happens if I miss my SHA contribution?",
     "answer": "Missing your monthly contribution can lead to suspension of benefits. To restore active coverage, you must clear all outstanding arrears. SHA encourages timely payments to ensure continuous access to healthcare and prevent interruptions during emergencies."},
    
    {"question": "What benefits are offered to pregnant women?",
     "answer": "SHA provides comprehensive maternity benefits including antenatal care, delivery (both normal and cesarean), and postnatal services. Pregnant women can access care at any empaneled facility without extra payment beyond their contributions. The aim is to promote safe motherhood and reduce maternal and child mortality rates."},
    
    {"question": "Are chronic illnesses covered under SHA?",
     "answer": "Yes. SHA covers treatment and management of chronic diseases such as diabetes, hypertension, cancer, and kidney failure. Members can access specialized clinics and long-term medication at empaneled facilities. SHA works to ensure affordability of life-saving treatments for chronic conditions."},
    
    {"question": "Can indigent or vulnerable persons get SHA coverage?",
     "answer": "Yes, indigent and vulnerable persons are fully covered through government-funded subsidies. The government identifies and registers these individuals under special categories to ensure no one is left behind in accessing essential health services."},
    
    {"question": "How can I access healthcare under SHA?",
     "answer": "Once registered, you can visit any SHA-empaneled healthcare provider. Present your SHA ID, membership card, or biometric data to verify eligibility. Services are then provided according to your coverage without direct payment at the point of care."},
    
    {"question": "How does SHA ensure quality healthcare services?",
     "answer": "SHA regularly monitors and audits all contracted healthcare providers to ensure compliance with national health standards. It evaluates service delivery, patient satisfaction, and outcomes to guarantee high-quality, patient-centered care. Providers failing to meet standards may face penalties or suspension."},
    
    {"question": "Can I change my healthcare provider?",
     "answer": "Yes. Members can change their preferred healthcare provider after a defined period, usually once every quarter. This can be done through the SHA online portal or by visiting an office. The process ensures members have the freedom to choose providers that best meet their healthcare needs."},
    
    {"question": "What happens when I retire or lose my job?",
     "answer": "When you retire or lose employment, you can continue contributing individually to keep your SHA membership active. SHA provides flexible payment options for retirees and self-employed individuals to prevent coverage loss during life transitions."},
    
    {"question": "What should I do if I lose my SHA ID or card?",
     "answer": "If you lose your SHA membership card or ID, report it immediately through the SHA portal or at the nearest office. You can request a replacement or access your digital ID online. Always keep your membership number and registered contact details up to date."},
    
    {"question": "How does SHA handle fraud or misuse of funds?",
     "answer": "SHA employs strict monitoring, audit trails, and digital verification to prevent fraud. Any cases of misuse or impersonation are investigated, and legal action is taken against offenders. Members are encouraged to report suspicious activities through official SHA hotlines."},
    
    {"question": "Can foreign nationals register under SHA?",
     "answer": "Yes, foreign nationals residing in Kenya with valid residence or work permits are eligible to register. They enjoy similar benefits as Kenyan citizens during their stay in the country. Contributions and verification processes may vary slightly based on visa type."},
    
    {"question": "Does SHA cover accidents and emergencies?",
     "answer": "Yes. SHA covers emergency medical care, including accidents and life-threatening conditions, at any empaneled or nearby accredited facility. Patients receive care first, and verification is done afterward to ensure no delay in life-saving services."},
    
    {"question": "Can I use SHA services outside Kenya?",
     "answer": "Currently, SHA benefits apply within Kenya. However, discussions are underway to establish partnerships for cross-border healthcare services in the East African region. Members traveling abroad should seek temporary health insurance for international coverage."},
    
    {"question": "What are empaneled healthcare providers?",
     "answer": "Empaneled providers are hospitals, clinics, and healthcare facilities that have been officially contracted by SHA to deliver healthcare services to members. These providers meet set quality and capacity standards. Members are encouraged to use these providers for guaranteed coverage and service quality."},
    
    {"question": "What are my responsibilities as an SHA member?",
     "answer": "Members are required to make regular contributions, keep their information updated, and use healthcare services responsibly. They should also report changes such as new dependents or employment status promptly. Compliance ensures smooth service delivery and continued access to healthcare."},
    
    {"question": "Where can I get more information about SHA?",
     "answer": "You can visit the official SHA website, follow their verified social media channels, or contact their customer service hotline. SHA offices across all counties also provide in-person assistance for registration, claims, and inquiries."},
]

faq_df = pd.DataFrame(faq_data)

# --------------------------------
# ⚙ TF-IDF Model
# --------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(faq_df["question"])

def chatbot_response(user_input):
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    index = similarities.argmax()
    score = similarities[0, index]
    if score < 0.3:
        return "🤔 I'm not sure about that,For more information  please contact SHA support at 0713889663  or for more help or visit the official SHA website for verified details."
    else:
        return faq_df.iloc[index]["answer"]

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
# 💬 Streamlit Chat Interface
# --------------------------------
st.markdown("""
    <h1 style='text-align: center;'>💬 Social Health Authority Chatbot</h1>
    <p style='text-align: center; color: gray;'>Your virtual assistant for SHA information and support in Kenya.</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar

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

# Input area
user_query = st.chat_input("Ask me something about SHA...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate bot response with typing animation
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = chatbot_response(user_query)
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.03)
        message_placeholder.markdown(full_response)

    # Save to session + CSV log
    st.session_state.chat_history.append({"user": user_query, "bot": response})
    log_chat(user_query, response)
