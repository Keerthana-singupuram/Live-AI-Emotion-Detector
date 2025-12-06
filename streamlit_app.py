import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from deepface import DeepFace

# --- CONFIGURATION ---
# This is required for the cloud to find your webcam
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- COLORS ---
emotion_colors = {
    'angry': (0, 0, 255), 'disgust': (0, 100, 0), 'fear': (128, 0, 128),
    'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (255, 255, 0),
    'neutral': (200, 200, 200)
}

# Load cascade once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionProcessor(VideoTransformerBase):
    def recv(self, frame):
        # Convert the frame to a numpy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror effect
        img = cv2.flip(img, 1)
        
        # 1. Convert to grayscale for Face Detection
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 2. Process Faces
        for (x, y, w, h) in faces:
            face_roi = img[y:y + h, x:x + w]
            try:
                # DeepFace Analysis
                # We use enforce_detection=False to stop crashes on blurry faces
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                
                # Colors & Drawing
                b, g, r = emotion_colors.get(emotion, (255, 255, 255))
                cv2.rectangle(img, (x, y), (x + w, y + h), (b, g, r), 2)
                
                # Text
                text_size, _ = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x, y - 30), (x + text_size[0] + 10, y), (b, g, r), -1)
                cv2.putText(img, emotion, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            except Exception:
                pass

        # Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- THE UI ---
st.title("🎥 Live AI Emotion Detector")
st.write("This app uses DeepFace to analyze emotions in real-time.")

# Start the webcam stream
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)