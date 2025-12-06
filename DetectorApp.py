import cv2
import numpy as np
from deepface import DeepFace
import gradio as gr

# --- COLORS ---
emotion_colors = {
    'angry': (0, 0, 255), 'disgust': (0, 100, 0), 'fear': (128, 0, 128),
    'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (255, 255, 0),
    'neutral': (200, 200, 200)
}

# --- SETUP ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(frame):
    if frame is None: return None
    
    # Mirror effect
    frame = cv2.flip(frame, 1)
    
    img_draw = frame.copy()
    gray_frame = cv2.cvtColor(img_draw, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        try:
            # Analyze
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            
            # Draw
            b, g, r = emotion_colors.get(emotion, (255, 255, 255))
            rgb_color = (r, g, b)
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), rgb_color, 2)
            
            # Text
            text_size, _ = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img_draw, (x, y - 30), (x + text_size[0] + 10, y), rgb_color, -1)
            cv2.putText(img_draw, emotion, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except:
            pass
            
    return img_draw

# --- VIDEO UI ---
with gr.Blocks(title="AI Emotion Video") as demo:
    gr.Markdown("# 🎥 Live AI Emotion Detector")
    
    with gr.Row():
        # Input: streaming=True enables the WebRTC connection
        input_cam = gr.Image(sources=["webcam"], streaming=True, label="Input")
        # Output: Shows the processed result
        output_cam = gr.Image(label="Output")

    # This trigger is what makes it a VIDEO. 
    # It runs every time a new frame arrives.
    input_cam.stream(fn=detect_emotion, inputs=input_cam, outputs=output_cam)


demo.launch()