import os
import tempfile
import av
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ---------------------------------------------------------------------------
# MediaPipe Tasks API setup
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmark = mp.tasks.vision.PoseLandmark
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
RunningMode = mp.tasks.vision.RunningMode
drawing_utils = mp.tasks.vision.drawing_utils
DrawingSpec = drawing_utils.DrawingSpec

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

# ---------------------------------------------------------------------------
# Color constants (Gold theme for overlays)
# ---------------------------------------------------------------------------
GOLD = (67, 168, 212)        # #D4A843 in BGR
GOLD_LIGHT = (102, 200, 235) # #EBC866 in BGR
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
DARK_BG = (23, 17, 14)       # #0E1117 in BGR

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calc_angle(a, b, c):
    """Return the angle (degrees) at vertex b given points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def get_landmark_coord(landmarks, idx, w, h):
    """Return pixel coordinates (x, y) for a landmark index."""
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]


def check_knee_over_toe(landmarks, w, h, side="left"):
    """Check if the knee extends too far past the toes."""
    if side == "left":
        knee_idx = PoseLandmark.LEFT_KNEE
        ankle_idx = PoseLandmark.LEFT_ANKLE
    else:
        knee_idx = PoseLandmark.RIGHT_KNEE
        ankle_idx = PoseLandmark.RIGHT_ANKLE
    knee_x = landmarks[knee_idx].x * w
    ankle_x = landmarks[ankle_idx].x * w
    return abs(knee_x - ankle_x) > 0.08 * w


def check_back_rounding(landmarks, w, h):
    """Check if the back is rounding (shoulder-hip-knee angle)."""
    left_shoulder = get_landmark_coord(landmarks, PoseLandmark.LEFT_SHOULDER, w, h)
    left_hip = get_landmark_coord(landmarks, PoseLandmark.LEFT_HIP, w, h)
    left_knee = get_landmark_coord(landmarks, PoseLandmark.LEFT_KNEE, w, h)
    trunk_angle = calc_angle(left_shoulder, left_hip, left_knee)
    return trunk_angle < 60


def draw_pose_on_image(img, landmarks):
    """Draw landmark points and skeleton connections on the image."""
    h, w, _ = img.shape
    connections = PoseLandmarksConnections.POSE_LANDMARKS
    for connection in connections:
        start_lm = landmarks[connection.start]
        end_lm = landmarks[connection.end]
        start_px = (int(start_lm.x * w), int(start_lm.y * h))
        end_px = (int(end_lm.x * w), int(end_lm.y * h))
        cv2.line(img, start_px, end_px, GOLD_LIGHT, 2, cv2.LINE_AA)

    for lm in landmarks:
        px = (int(lm.x * w), int(lm.y * h))
        cv2.circle(img, px, 4, GOLD, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Step 1: Shared analysis functions
# ---------------------------------------------------------------------------

def analyze_frame(landmarks, w, h):
    """Analyze a single frame's landmarks and return (knee_angle, warnings).

    Parameters
    ----------
    landmarks : list of NormalizedLandmark
    w, h : int – frame dimensions in pixels

    Returns
    -------
    knee_angle : float
    warnings : list[str]
    """
    left_hip = get_landmark_coord(landmarks, PoseLandmark.LEFT_HIP, w, h)
    left_knee = get_landmark_coord(landmarks, PoseLandmark.LEFT_KNEE, w, h)
    left_ankle = get_landmark_coord(landmarks, PoseLandmark.LEFT_ANKLE, w, h)
    right_hip = get_landmark_coord(landmarks, PoseLandmark.RIGHT_HIP, w, h)
    right_knee = get_landmark_coord(landmarks, PoseLandmark.RIGHT_KNEE, w, h)
    right_ankle = get_landmark_coord(landmarks, PoseLandmark.RIGHT_ANKLE, w, h)

    left_angle = calc_angle(left_hip, left_knee, left_ankle)
    right_angle = calc_angle(right_hip, right_knee, right_ankle)
    knee_angle = (left_angle + right_angle) / 2.0

    warnings: list[str] = []
    if check_knee_over_toe(landmarks, w, h, "left") or check_knee_over_toe(
        landmarks, w, h, "right"
    ):
        warnings.append("KNEE TOO FAR FORWARD!")
    if check_back_rounding(landmarks, w, h):
        warnings.append("KEEP YOUR BACK STRAIGHT!")

    return knee_angle, warnings


def update_squat_state(state: dict, knee_angle: float) -> str | None:
    """Update squat detection state and return a label when a rep completes.

    Parameters
    ----------
    state : dict with keys phase, min_angle, rep_count, full_count, half_count
    knee_angle : float – current knee angle in degrees

    Returns
    -------
    label : str or None – "FULL!", "HALF", "SHALLOW", or None if no rep completed
    """
    label = None
    if state["phase"] == "up" and knee_angle < 140:
        state["phase"] = "down"
        state["min_angle"] = knee_angle
    elif state["phase"] == "down":
        state["min_angle"] = min(state["min_angle"], knee_angle)
        if knee_angle > 160:
            state["rep_count"] += 1
            if state["min_angle"] <= 90:
                state["full_count"] += 1
                label = "FULL!"
            elif state["min_angle"] <= 120:
                state["half_count"] += 1
                label = "HALF"
            else:
                label = "SHALLOW"
            state["phase"] = "up"
            state["min_angle"] = 180.0
    return label


# ---------------------------------------------------------------------------
# Video Processor (refactored to use shared functions)
# ---------------------------------------------------------------------------
class SquatVideoProcessor(VideoProcessorBase):
    """Video processor for streamlit-webrtc real-time analysis."""

    def __init__(self):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

        self.rep_count = 0
        self.full_count = 0
        self.half_count = 0
        self.phase = "up"
        self.min_angle = 180.0
        self.angle_history: list[float] = []
        self.frame_idx = 0
        self.current_angle = 180.0
        self.current_label = ""
        self.warnings: list[str] = []

    def _get_state(self) -> dict:
        return {
            "phase": self.phase,
            "min_angle": self.min_angle,
            "rep_count": self.rep_count,
            "full_count": self.full_count,
            "half_count": self.half_count,
        }

    def _set_state(self, state: dict):
        self.phase = state["phase"]
        self.min_angle = state["min_angle"]
        self.rep_count = state["rep_count"]
        self.full_count = state["full_count"]
        self.half_count = state["half_count"]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp_ms += 33
        results = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]
            draw_pose_on_image(img, landmarks)

            knee_angle, warnings = analyze_frame(landmarks, w, h)
            self.current_angle = knee_angle
            self.frame_idx += 1
            self.angle_history.append(knee_angle)
            self.warnings = warnings

            state = self._get_state()
            label = update_squat_state(state, knee_angle)
            self._set_state(state)
            if label:
                self.current_label = label

            # --- Overlay drawing ---
            angle_text = f"Knee Angle: {int(knee_angle)} deg"
            cv2.putText(
                img, angle_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, GOLD, 2, cv2.LINE_AA,
            )

            rep_text = f"Reps: {self.rep_count}  (Full: {self.full_count} / Half: {self.half_count})"
            cv2.putText(
                img, rep_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA,
            )

            if self.current_label:
                label_color = GREEN if self.current_label == "FULL!" else GOLD
                cv2.putText(
                    img, self.current_label, (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, label_color, 4, cv2.LINE_AA,
                )

            for i, warn in enumerate(warnings):
                cv2.putText(
                    img, warn, (10, h - 40 - i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Video upload analysis pipeline
# ---------------------------------------------------------------------------

def process_uploaded_video(video_bytes, progress_bar):
    """Analyze an uploaded video file and return results dict.

    Returns
    -------
    dict with keys: rep_count, full_count, half_count, angle_history, keyframes
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    state = {
        "phase": "up",
        "min_angle": 180.0,
        "rep_count": 0,
        "full_count": 0,
        "half_count": 0,
    }
    angle_history: list[float] = []
    keyframes: list[np.ndarray] = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000.0 / fps))

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]
            knee_angle, _ = analyze_frame(landmarks, w, h)
            angle_history.append(knee_angle)

            prev_rep = state["rep_count"]
            label = update_squat_state(state, knee_angle)
            if label and state["rep_count"] > prev_rep:
                draw_pose_on_image(frame, landmarks)
                # Draw label on keyframe
                label_color = (0, 255, 0) if label == "FULL!" else (0, 165, 255)
                cv2.putText(
                    frame, f"Rep {state['rep_count']}: {label}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3, cv2.LINE_AA,
                )
                keyframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    landmarker.close()
    os.unlink(tmp_path)

    return {
        "rep_count": state["rep_count"],
        "full_count": state["full_count"],
        "half_count": state["half_count"],
        "angle_history": angle_history,
        "keyframes": keyframes,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Squat Police", layout="wide")

# --- Dark × Gold CSS Theme ---
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #1E3A5F 100%);
        border-right: 2px solid #D4A843;
    }

    /* Badge-style sidebar header */
    .sidebar-badge {
        text-align: center;
        padding: 1.2rem 0.8rem;
        margin-bottom: 0.5rem;
        border: 2px solid #D4A843;
        border-radius: 12px;
        background: linear-gradient(135deg, #0A1628, #162A46);
        box-shadow: 0 0 15px rgba(212, 168, 67, 0.15);
    }
    .sidebar-badge h1 {
        color: #D4A843 !important;
        font-size: 1.6rem !important;
        margin: 0 !important;
        letter-spacing: 2px;
    }
    .sidebar-badge p {
        color: #8899AA;
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Gold accent for headers */
    h1, h2, h3 {
        color: #D4A843 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #131A27;
        border: 1px solid #D4A843;
        border-radius: 8px;
        padding: 0.8rem;
    }
    div[data-testid="stMetric"] label {
        color: #8899AA !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #D4A843 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #D4A843 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #131A27;
        border: 1px solid #2A3A4F;
        border-radius: 8px 8px 0 0;
        color: #8899AA;
        padding: 0.5rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F !important;
        border-color: #D4A843 !important;
        color: #D4A843 !important;
    }

    /* Divider */
    hr {
        border-color: #2A3A4F !important;
    }

    /* Text */
    .main-subtitle {
        color: #8899AA;
        font-size: 1.1rem;
        margin-top: -0.8rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- session_state initialization ---
for key, default in [
    ("goal_full", 10),
    ("saved_rep_count", 0),
    ("saved_full_count", 0),
    ("saved_half_count", 0),
    ("saved_angle_history", []),
    ("saved_current_angle", 180.0),
    ("saved_warnings", []),
    ("session_active", False),
    ("upload_results", None),
    ("active_mode", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-badge">
            <h1>SQUAT POLICE</h1>
            <p>AI Form Inspector</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.session_state["goal_full"] = st.number_input(
        "Target Full Squats",
        min_value=1,
        max_value=100,
        value=st.session_state["goal_full"],
        step=1,
    )

    st.divider()
    stats_placeholder = st.empty()

# --- Main area ---
st.title("SQUAT POLICE")
st.markdown('<p class="main-subtitle">Stand in front of the camera or upload a video. We\'re watching your form.</p>', unsafe_allow_html=True)

# --- Tab-based mode switching ---
tab_realtime, tab_upload = st.tabs(["Real-time Camera", "Video Upload"])

# ===================== Real-time Camera Tab =====================
with tab_realtime:
    ctx = webrtc_streamer(
        key="squat-coach",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SquatVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    if ctx.video_processor:
        vp: SquatVideoProcessor = ctx.video_processor
        st.session_state["session_active"] = True
        st.session_state["active_mode"] = "live"
        st.session_state["saved_rep_count"] = vp.rep_count
        st.session_state["saved_full_count"] = vp.full_count
        st.session_state["saved_half_count"] = vp.half_count
        st.session_state["saved_angle_history"] = list(vp.angle_history)
        st.session_state["saved_current_angle"] = vp.current_angle
        st.session_state["saved_warnings"] = list(vp.warnings)
    elif st.session_state["session_active"] and st.session_state["active_mode"] == "live":
        st.session_state["session_active"] = False

# ===================== Video Upload Tab =====================
with tab_upload:
    uploaded_file = st.file_uploader("Upload a squat video (.mp4)", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button("Analyze Video", type="primary"):
            st.session_state["active_mode"] = "upload"
            progress_bar = st.progress(0.0, text="Analyzing video...")
            results = process_uploaded_video(uploaded_file.read(), progress_bar)
            progress_bar.progress(1.0, text="Analysis complete!")

            st.session_state["upload_results"] = results
            st.session_state["saved_rep_count"] = results["rep_count"]
            st.session_state["saved_full_count"] = results["full_count"]
            st.session_state["saved_half_count"] = results["half_count"]
            st.session_state["saved_angle_history"] = results["angle_history"]
            st.session_state["saved_current_angle"] = 180.0
            st.session_state["saved_warnings"] = []

            st.success(
                f"Analysis complete: {results['rep_count']} reps detected "
                f"({results['full_count']} full, {results['half_count']} half)"
            )

    # Keyframe gallery for upload mode
    if st.session_state.get("upload_results") and st.session_state["active_mode"] == "upload":
        keyframes = st.session_state["upload_results"].get("keyframes", [])
        if keyframes:
            st.subheader("Rep Keyframes")
            cols = st.columns(min(len(keyframes), 4))
            for i, kf in enumerate(keyframes):
                cols[i % len(cols)].image(kf, caption=f"Rep {i + 1}", use_container_width=True)

# --- Sidebar live stats ---
goal = st.session_state["goal_full"]
rep = st.session_state["saved_rep_count"]
full = st.session_state["saved_full_count"]
half = st.session_state["saved_half_count"]
cur_angle = st.session_state["saved_current_angle"]
warns = st.session_state["saved_warnings"]
mode = st.session_state["active_mode"]

with stats_placeholder.container():
    if mode:
        mode_label = "Live Session" if mode == "live" else "Video Analysis"
        st.caption(f"Mode: {mode_label}")

    progress = min(full / goal, 1.0) if goal > 0 else 0.0
    st.caption(f"Goal: {full} / {goal} Full Squats")
    st.progress(progress)
    if full >= goal and goal > 0:
        st.success("Goal reached!")

    st.metric("Total Reps", rep)
    col1, col2 = st.columns(2)
    col1.metric("Full", full)
    col2.metric("Half", half)
    if ctx.video_processor:
        st.metric("Current Angle", f"{int(cur_angle)}\u00b0")
        for w in warns:
            st.warning(w)

# ---------------------------------------------------------------------------
# Session Summary
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Session Summary")

angle_data = st.session_state["saved_angle_history"]

if angle_data:
    # --- Dark-themed angle chart ---
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#131A27")

    ax.plot(angle_data, color="#D4A843", linewidth=1.5, label="Knee angle")
    ax.axhline(y=90, color="#4CAF50", linestyle="--", alpha=0.7, label="Full squat (90\u00b0)")
    ax.axhline(y=120, color="#FF9800", linestyle="--", alpha=0.7, label="Half squat (120\u00b0)")
    ax.set_xlabel("Frame", color="#8899AA")
    ax.set_ylabel("Knee Angle (degrees)", color="#8899AA")
    ax.set_title("Knee Angle Over Time", color="#D4A843", fontsize=14)
    ax.legend(facecolor="#131A27", edgecolor="#2A3A4F", labelcolor="#CCCCCC")
    ax.tick_params(colors="#8899AA")
    for spine in ax.spines.values():
        spine.set_color("#2A3A4F")
    ax.set_ylim(0, 200)
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)

    # --- Stats ---
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Reps", rep)
    col_b.metric("Full Squats", full)
    col_c.metric("Half Squats", half)
    col_d.metric("Deepest Angle", f"{int(min(angle_data))}\u00b0")

    # --- Goal status ---
    if full >= goal:
        st.success(f"Mission accomplished! {goal} full squats completed. You're cleared, citizen.")
    else:
        st.info(f"You still owe {goal - full} full squats. Get back to work.")

    # --- Improvement advice (police tone, English) ---
    st.subheader("Inspector's Report")
    if rep == 0:
        st.info("No squats detected. Stand where we can see your full body, citizen.")
    else:
        full_ratio = full / rep if rep > 0 else 0.0
        if full_ratio >= 0.8:
            st.success(
                f"Impressive record: {full} out of {rep} reps were full squats. "
                "Your depth control is exemplary. Carry on."
            )
        elif full_ratio >= 0.5:
            st.warning(
                f"Mixed report: {full} out of {rep} reps reached full depth. "
                "Go deeper \u2014 thighs parallel to the floor is the standard."
            )
        else:
            st.error(
                f"Citation issued: Only {full}/{rep} reps were full squats. "
                "Work on mobility \u2014 wall squats can help you reach proper depth."
            )

        if min(angle_data) > 100:
            st.info(
                "Your deepest angle exceeded 100\u00b0. "
                "Try widening your stance or turning your toes out slightly to go lower."
            )

    # Keyframe gallery in summary (upload mode)
    if st.session_state.get("upload_results") and st.session_state["active_mode"] == "upload":
        keyframes = st.session_state["upload_results"].get("keyframes", [])
        if keyframes:
            st.subheader("Evidence: Rep Keyframes")
            cols = st.columns(min(len(keyframes), 4))
            for i, kf in enumerate(keyframes):
                cols[i % len(cols)].image(kf, caption=f"Rep {i + 1}", use_container_width=True)
else:
    st.info("Start a live session or upload a video to see your session summary here.")
