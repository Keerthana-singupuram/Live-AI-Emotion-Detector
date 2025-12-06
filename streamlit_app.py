import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from deepface import DeepFace

# --- CONFIGURATION (UPDATED FOR BETTER CONNECTION) ---
# We added multiple free STUN servers to fix the connection timeout
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]},
    ]}
)

# --- COLORS ---
emotion_colors = {
    'angry': (0, 0, 255),       # Red
    'disgust': (0, 100, 0),     # Dark Green
    'fear': (128, 0, 128),      # Purple
    'happy': (0, 255, 255),     # Yellow
    'sad': (255, 0, 0),         # Blue
    'surprise': (255, 255, 0),  # Cyan
    'neutral': (50, 50, 50)     # Dark Gray
}

# Load cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionProcessor(VideoTransformerBase):
    def recv(self, frame):
        # 1. Prepare Frame
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        height, width, _ = img.shape
        
        # 2. Detect Faces
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        current_emotion = "neutral"
        current_color = emotion_colors['neutral']

        # 3. Analyze
        for (x, y, w, h) in faces:
            face_roi = img[y:y + h, x:x + w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                current_emotion = result[0]['dominant_emotion']
                current_color = emotion_colors.get(current_emotion, (100, 100, 100))
                cv2.rectangle(img, (x, y), (x + w, y + h), current_color, 2)
            except Exception:
                pass

        # 4. Create Dashboard Panel
        panel_width = 300
        info_panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        info_panel[:] = current_color
        
        # Text
        text = current_emotion.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = int((panel_width - text_size[0]) / 2)
        text_y = int(height / 2)
        cv2.putText(info_panel, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # Combine
        combined_image = np.hstack((img, info_panel))
        return av.VideoFrame.from_ndarray(combined_image, format="bgr24")

# --- UI ---
st.set_page_config(page_title="AI Emotion Detector", layout="wide")
st.title("🎥 Live AI Emotion Dashboard")

webrtc_streamer(
    key="emotion-dashboard",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
