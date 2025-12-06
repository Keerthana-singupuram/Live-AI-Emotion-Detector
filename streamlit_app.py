import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from deepface import DeepFace

# --- CONFIGURATION ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- COLORS (BGR Format for OpenCV) ---
emotion_colors = {
    'angry': (0, 0, 255),       # Red
    'disgust': (0, 100, 0),     # Dark Green
    'fear': (128, 0, 128),      # Purple
    'happy': (0, 255, 255),     # Yellow
    'sad': (255, 0, 0),         # Blue
    'surprise': (255, 255, 0),  # Cyan
    'neutral': (50, 50, 50)     # Dark Gray
}

# Load cascade once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionProcessor(VideoTransformerBase):
    def recv(self, frame):
        # 1. Prepare the Frame
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effect
        
        # Get dimensions
        height, width, _ = img.shape
        
        # 2. Face Detection
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Default emotion state
        current_emotion = "neutral"
        current_color = emotion_colors['neutral']

        # 3. Analyze Faces
        for (x, y, w, h) in faces:
            face_roi = img[y:y + h, x:x + w]
            try:
                # DeepFace analysis
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                current_emotion = result[0]['dominant_emotion']
                current_color = emotion_colors.get(current_emotion, (100, 100, 100))
                
                # Draw Box around face
                cv2.rectangle(img, (x, y), (x + w, y + h), current_color, 2)
            except Exception:
                pass

        # --- CREATE THE SIDE PANEL (The "Second Box") ---
        # Create a blank panel (300 pixels wide)
        panel_width = 300
        info_panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        
        # Fill panel with the emotion color
        info_panel[:] = current_color
        
        # Add Text to the Panel
        text = current_emotion.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Calculate text size to center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = int((panel_width - text_size[0]) / 2)
        text_y = int(height / 2)
        
        # Draw Text (Black color for contrast)
        cv2.putText(info_panel, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # 4. COMBINE (Stitch the video and panel together)
        combined_image = np.hstack((img, info_panel))

        return av.VideoFrame.from_ndarray(combined_image, format="bgr24")

# --- THE UI ---
st.set_page_config(page_title="AI Emotion Detector", layout="wide")

st.title("🎥 Live AI Emotion Dashboard")
st.write("The Left box shows your camera. The Right box shows the Live AI Result.")

# Start the webcam stream
webrtc_streamer(
    key="emotion-dashboard",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
