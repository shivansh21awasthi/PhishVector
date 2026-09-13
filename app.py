import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="PhishVector", page_icon="🛡️", layout="centered")

# Sidebar for portfolio branding
st.sidebar.markdown("### 🛡️ PhishVector")
st.sidebar.info("An end-to-end Machine Learning REST API for real-time cyber threat detection.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Developer:** Shivansh Awasthi")
st.sidebar.markdown("**Branch:** CSE")

# Main UI
st.title("Email Threat Scanner")
st.write("Paste the contents of a suspicious email below to analyze it for phishing attempts.")

# Input area
email_text = st.text_area("Message Content", height=200, placeholder="e.g., URGENT: Verify your account details...")

# Scan button
if st.button("Scan for Threats", type="primary"):
    if email_text.strip():
        with st.spinner("Analyzing linguistic patterns..."):
            try:
                # Send the text to your local FastAPI server
                response = requests.post("https://phishvector.onrender.com/predict", json={"text": email_text})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result["prediction"] == "Phishing/Spam Detected":
                        st.error("🚨 **THREAT DETECTED:** This email exhibits strong phishing characteristics.")
                    else:
                        st.success("✅ **SAFE:** No malicious patterns detected in this email.")
                else:
                    st.error("Error communicating with the PhishVector API.")
            
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ Cannot connect to the API. Make sure your FastAPI uvicorn server is running in another terminal!")
    else:
        st.info("Please enter some text to scan.")