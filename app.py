# 1. API Key from Streamlit Secrets or Input
    try:
        # Tries to read the key from the secrets file (for deployment on Streamlit Cloud)
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key loaded from deployment secrets.")
    except (KeyError, AttributeError):
        # Fallback for local testing or if secrets are not set
        api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="The API Key is required to call the AI model for summarization locally or before setting up secrets."
        )
