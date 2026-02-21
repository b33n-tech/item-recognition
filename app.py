import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Object Recognition", layout="centered")

st.title("👁 Reconnaissance d'objet")

st.write("Détection : Visage / Main / Inconnu")

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

        self.hand_detector = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detected_something = False

        # -------------------------
        # VISAGES
        # -------------------------
        face_results = self.face_detector.process(rgb)

        if face_results.detections:
            detected_something = True
            for detection in face_results.detections:
                mp_draw.draw_detection(img, detection)

            cv2.putText(
                img,
                "VISAGE",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # -------------------------
        # MAINS
        # -------------------------
        hand_results = self.hand_detector.process(rgb)

        if hand_results.multi_hand_landmarks:
            detected_something = True
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            cv2.putText(
                img,
                "MAIN",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

        # -------------------------
        # OBJET INCONNU
        # -------------------------
        if not detected_something:
            cv2.putText(
                img,
                "OBJET NON IDENTIFIE",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="object-recognition",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
