import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="Eye Detector", layout="centered")
st.title("🎯 Eye Detector")
st.write("Bouge un objet dans la cible pour perdre.")

TARGET_RADIUS = 70
MOTION_THRESHOLD = 800

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=25
        )

    def transform(self, frame):  # 'transform' au lieu de 'recv'
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        center = (w // 2, h // 2)

        target_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(target_mask, center, TARGET_RADIUS, 255, -1)

        fg_mask = self.bg_subtractor.apply(img)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        target_motion = cv2.bitwise_and(thresh, thresh, mask=target_mask)
        motion_pixels = np.sum(target_motion == 255)

        cv2.circle(img, center, TARGET_RADIUS, (0, 0, 255), 2)

        if motion_pixels > MOTION_THRESHOLD:
            cv2.putText(
                img,
                "PERDU",
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="eye-detector",
    video_transformer_factory=VideoProcessor,  # paramètre mis à jour
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_transform=True  # améliore les performances
)

---
