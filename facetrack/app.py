import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
import pandas as pd

# Load trained model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

# Directory to save attendance
ATTENDANCE_DIR = "attendance"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Streamlit UI
st.title("📸 Face Recognition Attendance System")
st.markdown("---")

# Session selection
session = st.selectbox("Select Session", ["FN", "AN"])
today = datetime.now().strftime("%Y-%m-%d")
filename = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}_{session}.csv")

# Initialize attendance set
if "attendance_set" not in st.session_state:
    st.session_state.attendance_set = set()

# Start attendance button
start = st.button("▶ Start Attendance")
stop = st.button("⏹ Stop Attendance")

if start:
    st.info("Starting webcam...")

    # Start webcam using DirectShow (better for laptops)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
        st.stop()

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠ Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)
            if conf < 70:
                name = labels[id_]
                st.session_state.attendance_set.add(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        stframe.image(frame, channels="BGR")

        # If stop is clicked
        if stop:
            cap.release()
            cv2.destroyAllWindows()
            break

# Save attendance
if stop:
    if st.session_state.attendance_set:
        df = pd.DataFrame({"Name": list(st.session_state.attendance_set)})
        df["Date"] = today
        df["Session"] = session
        df["Status"] = "Present"

        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=["Name"], keep="last")
        else:
            combined_df = df

        combined_df.to_csv(filename, index=False)
        st.success(f"✅ Attendance saved in {filename}.")
        st.dataframe(combined_df)
    else:
        st.warning("⚠ No faces recognized. Attendance not saved.")