import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pytz

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "Database")
FOOTAGE_DIR = os.path.join(BASE_DIR, "Live Footage Images")

EMAIL_FROM = st.secrets["EMAIL"]["EMAIL_FROM"]
EMAIL_PASSWORD = st.secrets["EMAIL"]["EMAIL_PASSWORD"]

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(FOOTAGE_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
from datetime import datetime
import pytz  # Add this import at the top of your file

def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))
    return face

def send_email(to_email, student_name):
    msg = EmailMessage()
    msg["Subject"] = "Student Arrival Notification"
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email

    # ---------------- TIME FIX FOR IST ----------------
    india_tz = pytz.timezone("Asia/Kolkata")
    time_now = datetime.now(india_tz).strftime("%H:%M:%S")
    # -----------------------------------------------

    msg.set_content(
        f"Dear Parent,\n\nYour child {student_name} has arrived at school safely at {time_now}.\n\nRegards,\nSchool Administration"
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI-Enabled Facial Recognition Attendance System")
st.title("AI-Enabled Facial Recognition Attendance System")

tab1, tab2 = st.tabs(["📂 Student Database", "🎥 Live Footage Verification"])

# ---------------- DATABASE TAB ----------------
with tab1:
    st.header("Add Student to Database")

    name = st.text_input("Student Name")
    parent_email = st.text_input("Parent Email")
    uploaded_images = st.file_uploader(
        "Upload Student Images (1–3 clear photos)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Save Student"):
        if not name or not parent_email or not uploaded_images:
            st.error("Please fill all fields.")
        else:
            student_dir = os.path.join(DATABASE_DIR, name)
            os.makedirs(student_dir, exist_ok=True)

            faces = []
            saved = 0
            for img_file in uploaded_images:
                img = Image.open(img_file).convert("RGB")
                img_np = np.array(img)
                face = preprocess_face(img_np)

                if face is not None:
                    path = os.path.join(student_dir, f"{saved}.jpg")
                    cv2.imwrite(path, face)
                    faces.append(face)
                    saved += 1

            if saved == 0:
                st.error("No clear face detected. Try another image.")
            else:
                # Save parent email
                with open(os.path.join(student_dir, "email.txt"), "w") as f:
                    f.write(parent_email)

                # Train LBPH recognizer for this student
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                labels = np.array([0]*len(faces))  # all faces get label 0 for this student
                recognizer.train(faces, labels)
                recognizer.save(os.path.join(student_dir, "recognizer.yml"))

                st.success(f"{saved} image(s) saved and recognizer trained for {name}")

# ---------------- LIVE FOOTAGE TAB ----------------
with tab2:
    st.header("Verify Student from Footage Image")

    footage_image = st.file_uploader(
        "Upload Footage Image",
        type=["jpg", "jpeg", "png"]
    )

    if footage_image:
        img = Image.open(footage_image).convert("RGB")
        img_np = np.array(img)
        face = preprocess_face(img_np)

        if face is None:
            st.error("Face not detected in image.")
        else:
            best_match = None
            best_confidence = float('inf')
            best_email = None

            for student in os.listdir(DATABASE_DIR):
                student_dir = os.path.join(DATABASE_DIR, student)
                recognizer_path = os.path.join(student_dir, "recognizer.yml")
                if not os.path.exists(recognizer_path):
                    continue

                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(recognizer_path)

                label, confidence = recognizer.predict(face)
                # lower confidence = better match
                if confidence < best_confidence:
                    best_confidence = confidence
                    best_match = student
                    email_path = os.path.join(student_dir, "email.txt")
                    if os.path.exists(email_path):
                        best_email = open(email_path).read()

            # Threshold for confidence (you may tweak 50-80 based on testing)
            if best_confidence < 70:
                st.success(f"Student Identified: {best_match}")
                if best_email:
                    send_email(best_email, best_match)
                    st.info("Notification email sent.")
            else:
                st.warning("No matching student found.")
