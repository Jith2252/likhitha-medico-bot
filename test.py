import os
import streamlit as st
from openai import OpenAI
import io
import base64
import tempfile

# Configure the page with custom branding
st.set_page_config(
    page_title="Likhitha Medico-Bot 💊",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/jith2252/likhitha-medico-bot',
        'Report a bug': 'https://github.com/jith2252/likhitha-medico-bot/issues',
        'About': """
        ## Likhitha Medico-Bot 💊🩺
        
        **Your AI Medical Assistant**
        
        This application provides AI-powered medical information and assistance.
        Built with ❤️ using Streamlit and OpenAI.
        
        **Features:**
        - Medical Q&A
        - File upload support (documents, images, audio)
        - Voice input/output
        - Multi-format support
        
        **Disclaimer:** This is for informational purposes only. 
        Always consult qualified healthcare professionals for medical advice.
        """
    }
)
try:
    import pdfkit
except ImportError:
    pdfkit = None
try:
    import speech_recognition as sr
    from gtts import gTTS
except ImportError:
    sr = None
    gTTS = None
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Voice input/output support functions
def process_audio_file(audio_file):
    """Process uploaded audio file for speech recognition"""
    if sr is None:
        st.error("SpeechRecognition is not installed.")
        return ""
    
    try:
        recognizer = sr.Recognizer()
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Process the audio file
        with sr.AudioFile(tmp_file_path) as source:
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio)
            os.unlink(tmp_file_path)  # Clean up temp file
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try again with clearer audio.")
            os.unlink(tmp_file_path)
            return ""
        except sr.RequestError as e:
            st.error(f"Error with speech recognition service: {e}")
            os.unlink(tmp_file_path)
            return ""
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return ""

def text_to_speech(text):
    if gTTS is None:
        st.warning("gTTS is not installed.")
        return
    try:
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")


st.title("Likhitha Medico-Bot💊🩺")

# Hide Streamlit branding and customize appearance
st.markdown("""
<style>
/* Hide Streamlit branding but keep sidebar toggle */
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hide main menu items but keep sidebar toggle visible */
.stApp > header[data-testid="stHeader"] {
    display: none;
}

/* Keep sidebar toggle button visible */
button[data-testid="baseButton-header"] {
    display: block !important;
    visibility: visible !important;
}

/* Show sidebar toggle when sidebar is collapsed */
.stApp > div[data-testid="stSidebar"] > div > div > button {
    display: block !important;
    visibility: visible !important;
}

/* Ensure sidebar toggle is always accessible */
div[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
}

/* Hide "Made with Streamlit" */
.viewerBadge_container__1QSob {
    display: none !important;
}

/* Hide deploy button and other menu items, but not sidebar toggle */
.stDeployButton {
    display: none !important;
}

/* Custom header styling */
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Custom footer */
.custom-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(240, 242, 246, 0.9);
    color: #262730;
    text-align: center;
    padding: 10px;
    font-size: 0.8rem;
    z-index: 999;
}

/* Ensure sidebar controls remain functional */
.stSidebar button {
    display: block !important;
    visibility: visible !important;
}

/* Show sidebar toggle icon in top-left corner */
.stApp > div[data-testid="stSidebar"] {
    position: relative;
}

/* Make sure the collapse/expand button is always visible */
button[kind="header"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# Apply custom theme CSS based on user preference
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

# Custom CSS for theme enhancement
if st.session_state.get("theme_mode") == "light":
    st.markdown("""
    <style>
    /* AGGRESSIVE LIGHT THEME - Override everything */
    *, *::before, *::after {
        color: #212529 !important;
    }
    
    /* Main app structure */
    .stApp {
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    .main .block-container {
        background-color: #ffffff !important;
    }
    .stSidebar {
        background-color: #f8f9fa !important;
    }
    .stSidebar .sidebar-content {
        background-color: #f8f9fa !important;
    }
    
    /* Target ALL possible containers that could be dark */
    div, section, article, main, aside, nav, header, footer {
        background-color: inherit !important;
    }
    
    /* Specifically target known dark containers */
    div[data-testid="stFileUploader"], 
    div[data-testid="stChatInput"],
    div[data-baseweb="file-uploader"],
    section[data-testid="stFileUploader"],
    section[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    
    /* All file uploader related elements */
    .stFileUploader, .stFileUploader *, 
    .stFileUploader > div, .stFileUploader > div > div,
    .stFileUploader > div > div > div {
        background-color: #ffffff !important;
        color: #212529 !important;
        border-color: #ced4da !important;
    }
    
    /* All chat input related elements */
    .stChatInput, .stChatInput *,
    .stChatInput > div, .stChatInput > div > div,
    .stChatInput input, .stChatInput textarea {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Input fields for light theme */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Buttons for light theme */
    .stButton > button {
        background-color: #007bff !important;
        color: white !important;
        border: 1px solid #007bff !important;
    }
    .stButton > button:hover {
        background-color: #0056b3 !important;
        border-color: #0056b3 !important;
    }
    
    /* Chat messages for light theme */
    .stChatMessage {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        color: #212529 !important;
    }
    
    /* Override any inline styles that might be dark */
    [style*="background-color: rgb(14, 17, 23)"] {
        background-color: #ffffff !important;
    }
    [style*="background-color: rgb(38, 39, 48)"] {
        background-color: #ffffff !important;
    }
    [style*="background-color: rgb(49, 51, 63)"] {
        background-color: #ffffff !important;
    }
    
    /* Bottom area where chat input lives */
    .stBottom, .stBottom * {
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    
    /* Any remaining elements */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    /* Dark theme customizations */
    .stApp {
        background-color: #0e1117 !important;
    }
    .main .block-container {
        background-color: #0e1117 !important;
    }
    .stSidebar {
        background-color: #262730 !important;
    }
    .stSidebar .sidebar-content {
        background-color: #262730 !important;
    }
    /* Input fields for dark theme */
    .stTextInput > div > div > input {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #464646 !important;
    }
    .stSelectbox > div > div > select {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #464646 !important;
    }
    .stTextArea > div > div > textarea {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #464646 !important;
    }
    /* Buttons for dark theme */
    .stButton > button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: 1px solid #ff4b4b !important;
    }
    .stButton > button:hover {
        background-color: #ff2b2b !important;
        border-color: #ff2b2b !important;
    }
    /* Chat messages for dark theme */
    .stChatMessage {
        background-color: #262730 !important;
        border: 1px solid #464646 !important;
        color: #fafafa !important;
    }
    /* File uploader for dark theme */
    .stFileUploader {
        background-color: #262730 !important;
        border: 2px dashed #464646 !important;
    }
    .stFileUploader > div {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    .stFileUploader label {
        color: #fafafa !important;
    }
    /* Chat input for dark theme */
    .stChatInput {
        background-color: #262730 !important;
    }
    .stChatInput > div > div {
        background-color: #262730 !important;
        border: 1px solid #464646 !important;
    }
    .stChatInput input {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    /* Header and navigation for dark theme */
    .stApp > header {
        background-color: #0e1117 !important;
    }
    .stToolbar {
        background-color: #0e1117 !important;
    }
    /* General text color for dark theme */
    .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    /* Headers and titles */
    .stTitle, .stHeader, .stSubheader {
        color: #fafafa !important;
    }
    </style>
    """, unsafe_allow_html=True)



# Unified file upload support for documents, images, and audio
st.markdown("---")
st.subheader("📁 File Upload & Voice Input")

# Single file uploader for all file types
uploaded_file = st.file_uploader(
    "Upload a file for analysis or voice input:",
    type=["jpg", "jpeg", "png", "pdf", "txt", "wav", "mp3", "m4a"],
    help="• Documents/Images: JPG, PNG, PDF, TXT for reference/Q&A\n• Audio: WAV, MP3, M4A for speech-to-text conversion"
)

uploaded_file_text = None
if uploaded_file is not None:
    file_type = uploaded_file.type
    
    # Handle image files
    if file_type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.success("Image uploaded successfully for reference.")
    
    # Handle PDF files
    elif file_type == "application/pdf" and PyPDF2 is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            uploaded_file_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            st.success("PDF uploaded and text extracted for Q&A.")
        except Exception as e:
            st.error(f"Failed to extract PDF text: {e}")
    
    # Handle text files
    elif file_type == "text/plain":
        uploaded_file_text = uploaded_file.read().decode("utf-8")
        st.success("Text file uploaded for Q&A.")
    
    # Handle audio files
    elif file_type.startswith("audio") or uploaded_file.name.endswith(('.wav', '.mp3', '.m4a')):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Audio Preview:**")
            st.audio(uploaded_file, format="audio/wav")
        
        with col2:
            st.markdown("**Convert to Text:**")
            if st.button("🎙️ Convert Audio to Text"):
                with st.spinner("Processing audio..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    recognized_text = process_audio_file(uploaded_file)
                    if recognized_text:
                        st.session_state["voice_input"] = recognized_text
                        st.success(f"Converted text: {recognized_text}")
                        st.rerun()
    
    else:
        st.info("File type not supported. Please upload: Images (JPG, PNG), Documents (PDF, TXT), or Audio (WAV, MP3, M4A).")

# Display voice input if available
if "voice_input" in st.session_state:
    st.success(f"🎤 Voice input: {st.session_state['voice_input']}")

# Text-to-speech output section
st.markdown("**🔊 Audio Output:**")
if gTTS is not None and st.session_state.get("last_assistant_response"):
    if st.button("🔊 Play Last Assistant Response"):
        with st.spinner("Generating audio..."):
            text_to_speech(st.session_state["last_assistant_response"])
else:
    st.info("Text-to-speech will be available after getting a response from the assistant.")

# Try to load API key from environment variable or session state
if "api_key" not in st.session_state:
    env_api_key = os.getenv("api_key", "")
    if env_api_key:
        st.session_state["api_key"] = env_api_key

# If not found in environment or session, ask the user
if "api_key" not in st.session_state or not st.session_state["api_key"]:
    api_key_input = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key_input:
        st.session_state["api_key"] = api_key_input

api_key = st.session_state.get("api_key", "")

with st.sidebar:
    st.header("Settings & Info")
    
    # Theme toggle with visual feedback
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"
    
    st.markdown("**Theme Selection:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌙 Dark", disabled=(st.session_state["theme_mode"] == "dark")):
            st.session_state["theme_mode"] = "dark"
            st.rerun()
    
    with col2:
        if st.button("☀️ Light", disabled=(st.session_state["theme_mode"] == "light")):
            st.session_state["theme_mode"] = "light"
            st.rerun()
    
    # Show current theme status
    current_theme = st.session_state["theme_mode"]
    if current_theme == "dark":
        st.success("🌙 Dark theme active")
    else:
        st.info("☀️ Light theme active")
    
    # Note about Streamlit theme
    st.caption("💡 For best results, set your browser/system to the same theme, or use Streamlit's built-in theme toggle in the top-right menu.")
    
    st.markdown("---")
    # Model selection
    model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_options[0]
    selected_model = st.selectbox("Select OpenAI Model", model_options, index=model_options.index(st.session_state["openai_model"]))
    st.session_state["openai_model"] = selected_model
    # Session info
    st.write(f"**Current Model:** {st.session_state['openai_model']}")
    st.write(f"**Messages in Session:** {len(st.session_state.get('messages', []))}")
    # Log out / Change API Key button
    if st.button("Log out / Change API Key"):
        st.session_state.pop("api_key", None)
        st.rerun()
    # Download chat history
    if st.session_state.get("messages"):
        chat_str = ""
        for m in st.session_state["messages"]:
            chat_str += f"{m['role'].capitalize()}: {m['content']}\n\n"
        st.download_button("Download Chat History", data=chat_str, file_name="chat_history.txt")
        # Export as PDF
        if pdfkit:
            if st.button("Export Chat as PDF"):
                # Create a simple HTML for the chat
                html = "<h2>Chat History</h2>"
                for m in st.session_state["messages"]:
                    html += f"<b>{m['role'].capitalize()}:</b> {m['content']}<br><br>"
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    pdfkit.from_string(html, tmp_pdf.name)
                    tmp_pdf.seek(0)
                    b64 = base64.b64encode(tmp_pdf.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="chat_history.pdf">Download PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("Install pdfkit and wkhtmltopdf to enable PDF export.")

if api_key:
    client = OpenAI(api_key=api_key)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


    # Display existing chat history or welcome/help message
    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown("""
                👋 **Welcome to Likhitha Medico-Bot!**
                
                Ask me any medical or healthcare-related question. You can also upload an image for reference, or use the sidebar for more options.
                
                **Tip:** Try questions like:
                - What are the symptoms of diabetes?
                - How can I lower my blood pressure?
                - What is a healthy diet for heart patients?
            """)
    else:
        for idx, message in enumerate(st.session_state["messages"]):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Feedback/rating for assistant responses
                if message["role"] == "assistant":
                    if "feedback" not in st.session_state:
                        st.session_state["feedback"] = {}
                    fb = st.session_state["feedback"].get(idx)
                    col1, col2 = st.columns([1,1])
                    with col1:
                        if st.button("👍", key=f"up_{idx}"):
                            st.session_state["feedback"][idx] = "up"
                    with col2:
                        if st.button("👎", key=f"down_{idx}"):
                            st.session_state["feedback"][idx] = "down"
                    if fb == "up":
                        st.success("You rated this response as helpful.")
                    elif fb == "down":
                        st.error("You rated this response as not helpful.")

    # Clear Chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

    # Get user input
    # Use voice input if available, else chat input
    # Quick-access medical questions
    st.markdown("**Quick Medical Questions:**")
    quick_questions = [
        "What are the symptoms of diabetes?",
        "How can I lower my blood pressure?",
        "What is a healthy diet for heart patients?",
        "What are the side effects of paracetamol?",
        "How to manage stress for better health?"
    ]
    qcols = st.columns(len(quick_questions))
    for i, q in enumerate(quick_questions):
        if qcols[i].button(q):
            st.session_state["quick_question"] = q
            st.rerun()

    # Check for different input sources
    user_input = None
    
    # Check for voice input first
    if "voice_input" in st.session_state:
        user_input = st.session_state.pop("voice_input")
    
    # Check for quick question input
    elif "quick_question" in st.session_state:
        user_input = st.session_state.pop("quick_question")
    
    # Always show the chat input box
    chat_input = st.chat_input("What is up?")
    if chat_input:
        user_input = chat_input
    if user_input:

        # System prompt for medical expertise, with file context if available
        if uploaded_file_text:
            system_content = (
                "You are a highly knowledgeable and ethical medical assistant. "
                "The user has uploaded the following document for reference. Use it to answer their questions if relevant. "
                "Document content:\n" + uploaded_file_text[:3000] + "\n(End of document excerpt)\n"  # limit context for token safety
                "Only answer medical and healthcare related queries. If a user asks a non-medical question, politely refuse and explain you can only answer medical/healthcare questions."
            )
        else:
            system_content = (
                "You are a highly knowledgeable and ethical medical assistant. "
                "Only answer medical and healthcare related queries. If a user asks a non-medical question, politely refuse and explain you can only answer medical/healthcare questions."
            )
        system_prompt = {"role": "system", "content": system_content}
        modified_input = f"{user_input}"

        # Save original user input for UI
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant response (streaming) with loading spinner
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            with st.spinner("Assistant is typing..."):
                # Build messages for OpenAI (prepend system prompt)
                messages_for_api = [system_prompt]
                for m in st.session_state["messages"]:
                    messages_for_api.append(m)

                # Collect token usage and cost
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
                response_obj = None
                # Use non-streaming for token/cost info
                try:
                    response_obj = client.chat.completions.create(
                        model=st.session_state["openai_model"],
                        messages=messages_for_api,
                        stream=False,
                    )
                    full_response = response_obj.choices[0].message.content
                    usage = response_obj.usage.to_dict() if hasattr(response_obj, "usage") else usage
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")
                    full_response = "[Error: Could not get response]"
                response_placeholder.markdown(full_response)

                # Estimate cost (as of Oct 2025, update as needed)
                model_prices = {
                    "gpt-3.5-turbo": 0.0005,  # per 1K tokens
                    "gpt-4": 0.01,
                    "gpt-4o": 0.005
                }
                price_per_1k = model_prices.get(st.session_state["openai_model"], 0.001)
                usage["cost"] = (usage.get("total_tokens", 0) / 1000) * price_per_1k

        # Save assistant response and usage
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
        st.session_state["last_assistant_response"] = full_response
        if "token_usage" not in st.session_state:
            st.session_state["token_usage"] = []
        st.session_state["token_usage"].append(usage)

else:
    st.warning("Please set the `OPENAI_API_KEY` environment variable or enter it above.")

# Show token usage/cost in sidebar
with st.sidebar:
    if st.session_state.get("token_usage"):
        total_tokens = sum(u.get("total_tokens", 0) for u in st.session_state["token_usage"])
        total_cost = sum(u.get("cost", 0.0) for u in st.session_state["token_usage"])
        st.markdown(f"**Total tokens used:** {total_tokens}")
        st.markdown(f"**Estimated cost:** ${total_cost:.4f}")

