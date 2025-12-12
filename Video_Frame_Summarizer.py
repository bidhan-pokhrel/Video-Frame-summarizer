import streamlit as st
import cv2
import os
from PIL import Image
from google import genai
from google.genai import types
import tempfile
import time # Used for displaying status messages

# --- Configuration ---
OUTPUT_DIR = "extracted_frames_temp" # Temporary directory for frames
MODEL_NAME = "gemini-2.5-flash"
FRAME_EXTRACTION_INTERVAL = 30 # Default to 1 frame per second at 30fps

# --- Helper Functions (Modified for Streamlit/Temp Files) ---

def extract_frames(video_file_path, output_dir, interval):
    """
    Extracts frames from a video file at a specified interval.
    Returns a list of paths to the extracted image files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file.")
        return []

    frame_count = 0
    extracted_frames = []
    status_bar = st.progress(0, text="Extracting frames...")
    
    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            # Save frame (OpenCV uses BGR, save to file as JPEG)
            cv2.imwrite(frame_filename, frame) 
            extracted_frames.append(frame_filename)
        
        frame_count += 1
        
        # Update progress bar
        if total_frames > 0:
            progress = min(1.0, frame_count / total_frames)
            status_bar.progress(progress, text=f"Extracted {len(extracted_frames)} key frames.")


    cap.release()
    status_bar.empty() # Clear the progress bar after completion
    return extracted_frames

def summarize_frames(image_paths, api_key):
    """
    Sends images to the Gemini API for summarization.
    """
    if not image_paths:
        st.warning("No frames were extracted to summarize.")
        return "No summary generated."

    st.info(f"Analyzing {len(image_paths)} key frames with the Gemini model...")
    
    try:
        # Initialize client with the API key provided by the user
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client. Check your API Key. Error: {e}")
        return "API Client initialization failed."

    # The prompt tells the model exactly what you want
    prompt = (
        "Analyze the following sequence of images extracted from a video. "
        "Provide a comprehensive summary of the video content. "
        "First, create a short paragraph (3-4 sentences). "
        "Second, provide a bulleted list of 5 key events or takeaways."
    )

    # Convert the local image paths into Part objects for the API
    image_parts = []
    for path in image_paths:
        try:
            # Use PIL to open the image and create a Part
            img = Image.open(path)
            image_parts.append(img)
        except Exception as e:
            st.error(f"Could not open image file {path}: {e}")
            continue
    
    # Combine the prompt and the image parts into the contents list
    contents = [prompt] + image_parts
    
    # Use a spinner for the API call
    with st.spinner("🚀 Generating AI Summary... This may take a minute or two."):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents
            )
            return response.text
        except Exception as e:
            st.error(f"An error occurred during API call: {e}")
            return f"API summarization failed: {e}"

# --- Streamlit UI ---

st.set_page_config(
    page_title="Video-to-Summary AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Video Content Summarizer (Gemini AI)")
st.caption("Upload a video and let the Gemini model analyze its key frames to generate a summary.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # 1. API Key Input
    api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="The API Key is required to call the AI model for summarization."
    )
    
    # 2. Frame Interval Slider
    interval = st.slider(
        "Frame Extraction Interval (N)",
        min_value=10, max_value=120, value=FRAME_EXTRACTION_INTERVAL, step=10,
        help="Extract 1 frame every 'N' video frames. Use a higher number for longer videos (to reduce API cost/time)."
    )
    st.write(f"Sampling rate: Approximately 1 frame every {interval/30:.1f} seconds (assuming 30 FPS).")
    
    st.markdown("---")
    st.info("The application uses temporary files for video and extracted frames, which are automatically cleaned up.")

# --- Main Content ---

uploaded_file = st.file_uploader("Upload a Video File (.mp4, .mov, etc.)", type=['mp4', 'mov', 'avi'])

if uploaded_file and api_key:
    # Use a temporary directory/file to handle the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    st.video(video_path, format=uploaded_file.type, start_time=0)
    
    if st.button("Generate Summary"):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
            st.stop()
            
        # Use a temporary directory for frames, automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            
            st.subheader("1. Frame Extraction")
            
            # Extract frames
            extracted_frames = extract_frames(video_path, temp_dir, interval)
            
            if extracted_frames:
                st.success(f"Successfully extracted **{len(extracted_frames)}** key frames.")
                
                # Optionally display a few extracted frames
                st.markdown("---")
                st.subheader("Sampled Key Frames:")
                cols = st.columns(min(len(extracted_frames), 5))
                for i, frame_path in enumerate(extracted_frames[:5]):
                    cols[i].image(frame_path, caption=f"Frame {i+1}", use_column_width=True)
                st.markdown("---")
                
                # Generate summary
                st.subheader("2. AI Summarization")
                summary = summarize_frames(extracted_frames, api_key)
                
                # Display result
                st.markdown("---")
                st.subheader("✨ Final Video Content Summary")
                st.markdown(summary)
            
            else:
                st.error("Frame extraction failed. Please check the video file.")
    
    # Clean up the temporary video file after processing
    os.unlink(video_path)

elif uploaded_file and not api_key:
    st.warning("Please enter your **Gemini API Key** in the sidebar to enable summarization.")
