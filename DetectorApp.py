import cv2
import numpy as np
from deepface import DeepFace
import gradio as gr
import os
import time

# --- SETUP ---
# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create folder for flagged images
if not os.path.exists("flagged"):
    os.makedirs("flagged")

# --- EMOTION COLORS ---
emotion_colors = {
    'angry': (0, 0, 255), 'disgust': (0, 100, 0), 'fear': (128, 0, 128),
    'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (255, 255, 0),
    'neutral': (200, 200, 200)
}

def detect_emotion(frame):
    if frame is None: return None
    
    # Mirror and Process
    frame = cv2.flip(frame, 1)
    img_draw = frame.copy()
    gray_frame = cv2.cvtColor(img_draw, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            
            # Colors & Draw
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

def flag_output(image):
    if image is None: return "No image captured."
    timestamp = int(time.time())
    filename = f"flagged/emotion_{timestamp}.png"
    save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, save_img)
    return f"Saved: {filename}"

# --- UI ---
with gr.Blocks(title="AI Emotion Video") as demo:
    
    # 1. CENTERED TITLE
    # Changed <h2> to <h3> and added some style to make it look cleaner
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 12px;">
            <h1 style="margin-bottom: 0px;">🎥 Live AI Emotion Detector</h1>
            <h3 style="margin-top: 5px; font-weight: normal; color: #555;">A real-time face expression analysis</h3>
        </div>
    """)
    
    # 2. VIDEO BOXES
    with gr.Row():
        # Input: Webcam source
        input_cam = gr.Image(sources=["webcam"], streaming=True, label="Input Camera")
        
        # Output: interactive=False removes the "Upload" and "Drop Image" buttons
        output_cam = gr.Image(label="AI Output", interactive=False)

    # 3. BUTTONS BELOW
    with gr.Row():
        clear_btn = gr.Button("⏹️ Clear", variant="secondary")
        flag_btn = gr.Button("🚩 Save", variant="primary")
    
    # Hidden status box
    status = gr.Markdown(visible=True)

    # --- LOGIC ---
    # Start stream
    input_cam.stream(fn=detect_emotion, inputs=input_cam, outputs=output_cam)
    
    # Button actions
    flag_btn.click(fn=flag_output, inputs=output_cam, outputs=status)
    clear_btn.click(lambda: (None, None), inputs=None, outputs=[input_cam, output_cam])

if __name__ == "__main__":
    demo.launch()

    