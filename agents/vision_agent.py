"""
VisionAgent - Computer Vision Agent using OpenCV and MediaPipe.

This agent provides visual input capabilities for the AI assistant:
- Hand gesture detection and recognition
- Face detection and recognition
- Presence detection (is user in front of camera)
- Visual commands via gestures

All communication is via EventBus only - no direct agent calls.

Dependencies:
    pip install opencv-python mediapipe numpy

Supported Gestures:
    - THUMBS_UP: Approval, confirm action
    - THUMBS_DOWN: Disapproval, cancel
    - WAVE: Greeting, get attention (open palm moving)
    - STOP: Stop/pause (open palm, fingers up)
    - FIST: Closed fist
    - OPEN_PALM: Open hand, palm facing camera
    - PEACE: Peace/victory sign (index + middle fingers)
    - OK: Okay gesture (thumb + index circle)
    - POINT: Pointing gesture (index finger extended)
    - ONE through FIVE: Finger counting

Face Recognition:
    - Uses face_recognition library (dlib-based) for encoding
    - Stores known faces in data/faces/
    - Can learn new faces on command
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from multiprocessing import Process, Queue, Value
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# OpenCV and MediaPipe imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

# Optional face_recognition for advanced face recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None

from agents.base_agent import AgentCapability, BaseAgent
from bus.event_bus import EventBus
from schemas.events import (
    FaceDetectedEvent,
    FaceLostEvent,
    GestureDetectedEvent,
    IntentRecognizedEvent,
    PresenceChangedEvent,
    ShutdownRequestedEvent,
    VisionCommandEvent,
    VisionCommandResultEvent,
    VisionStartEvent,
    VisionStopEvent,
    VoiceOutputEvent,
)
from utils.logger import get_logger, get_agent_logger


# =============================================================================
# Gesture Definitions
# =============================================================================

class Gesture(Enum):
    """Supported hand gestures."""
    UNKNOWN = "unknown"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    WAVE = "wave"
    STOP = "stop"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    PEACE = "peace"
    OK = "ok"
    POINT = "point"
    ONE = "one"
    TWO = "two"
    THREE = "three"
    FOUR = "four"
    FIVE = "five"


# Gesture to action mapping for common gestures
GESTURE_ACTIONS = {
    Gesture.THUMBS_UP: "confirm",
    Gesture.THUMBS_DOWN: "cancel",
    Gesture.WAVE: "greeting",
    Gesture.STOP: "stop",
    Gesture.OPEN_PALM: "pause",
    Gesture.PEACE: "screenshot",
    Gesture.OK: "accept",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HandLandmarks:
    """Processed hand landmark data."""
    wrist: Tuple[float, float, float]
    thumb_tip: Tuple[float, float, float]
    thumb_ip: Tuple[float, float, float]
    thumb_mcp: Tuple[float, float, float]
    thumb_cmc: Tuple[float, float, float]
    index_tip: Tuple[float, float, float]
    index_dip: Tuple[float, float, float]
    index_pip: Tuple[float, float, float]
    index_mcp: Tuple[float, float, float]
    middle_tip: Tuple[float, float, float]
    middle_dip: Tuple[float, float, float]
    middle_pip: Tuple[float, float, float]
    middle_mcp: Tuple[float, float, float]
    ring_tip: Tuple[float, float, float]
    ring_dip: Tuple[float, float, float]
    ring_pip: Tuple[float, float, float]
    ring_mcp: Tuple[float, float, float]
    pinky_tip: Tuple[float, float, float]
    pinky_dip: Tuple[float, float, float]
    pinky_pip: Tuple[float, float, float]
    pinky_mcp: Tuple[float, float, float]


@dataclass
class DetectedFace:
    """Information about a detected face."""
    face_id: str
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: Dict[str, Tuple[int, int]]
    encoding: Optional[np.ndarray] = None
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    name: Optional[str] = None
    confidence: float = 0.0


# =============================================================================
# Gesture Recognizer
# =============================================================================

class GestureRecognizer:
    """
    Recognizes hand gestures from MediaPipe hand landmarks.
    
    Uses geometric analysis of finger positions to classify gestures.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self._gesture_history: List[Gesture] = []
        self._history_size = 5
        self._confidence_threshold = 0.7
    
    def recognize(self, landmarks: HandLandmarks, handedness: str = "Right") -> Tuple[Gesture, float]:
        """
        Recognize gesture from hand landmarks.
        
        Args:
            landmarks: Processed hand landmark positions
            handedness: 'Left' or 'Right' hand
        
        Returns:
            Tuple of (gesture, confidence)
        """
        # Get finger states
        fingers = self._get_finger_states(landmarks, handedness)
        
        # Check for specific gestures
        gesture, confidence = self._classify_gesture(fingers, landmarks)
        
        # Apply temporal smoothing
        self._gesture_history.append(gesture)
        if len(self._gesture_history) > self._history_size:
            self._gesture_history.pop(0)
        
        # Return most common gesture in history for stability
        if len(self._gesture_history) >= 3:
            from collections import Counter
            most_common = Counter(self._gesture_history).most_common(1)[0]
            if most_common[1] >= 3:
                return most_common[0], confidence
        
        return gesture, confidence
    
    def _get_finger_states(self, lm: HandLandmarks, handedness: str) -> Dict[str, bool]:
        """
        Determine which fingers are extended.
        
        Returns dict with keys: thumb, index, middle, ring, pinky
        """
        # Thumb: compare x position (different for left/right hand)
        if handedness == "Right":
            thumb_extended = lm.thumb_tip[0] < lm.thumb_ip[0]
        else:
            thumb_extended = lm.thumb_tip[0] > lm.thumb_ip[0]
        
        # Other fingers: compare y positions (tip above pip = extended)
        # Note: y increases downward in image coordinates
        index_extended = lm.index_tip[1] < lm.index_pip[1]
        middle_extended = lm.middle_tip[1] < lm.middle_pip[1]
        ring_extended = lm.ring_tip[1] < lm.ring_pip[1]
        pinky_extended = lm.pinky_tip[1] < lm.pinky_pip[1]
        
        return {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended,
        }
    
    def _classify_gesture(
        self, 
        fingers: Dict[str, bool], 
        lm: HandLandmarks
    ) -> Tuple[Gesture, float]:
        """Classify gesture based on finger states and positions."""
        
        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]
        
        extended_count = sum([thumb, index, middle, ring, pinky])
        
        # FIST: no fingers extended
        if extended_count == 0:
            return Gesture.FIST, 0.9
        
        # ONE: only index extended
        if index and not middle and not ring and not pinky and not thumb:
            return Gesture.ONE, 0.85
        
        # POINT: index extended, others curled
        if index and not middle and not ring and not pinky:
            return Gesture.POINT, 0.8
        
        # PEACE: index and middle extended
        if index and middle and not ring and not pinky:
            return Gesture.PEACE, 0.85
        
        # THREE: index, middle, ring extended
        if index and middle and ring and not pinky and not thumb:
            return Gesture.THREE, 0.85
        
        # FOUR: all except thumb
        if index and middle and ring and pinky and not thumb:
            return Gesture.FOUR, 0.85
        
        # FIVE / OPEN_PALM: all fingers extended
        if extended_count == 5:
            return Gesture.OPEN_PALM, 0.9
        
        # THUMBS_UP: only thumb extended, hand upright
        if thumb and not index and not middle and not ring and not pinky:
            # Check if thumb is pointing up (thumb tip above wrist)
            if lm.thumb_tip[1] < lm.wrist[1]:
                return Gesture.THUMBS_UP, 0.85
            else:
                return Gesture.THUMBS_DOWN, 0.85
        
        # OK: thumb and index touching, others extended
        thumb_index_dist = self._distance(lm.thumb_tip, lm.index_tip)
        if thumb_index_dist < 0.05 and middle and ring and pinky:
            return Gesture.OK, 0.8
        
        # STOP: palm facing camera, all fingers up
        if extended_count >= 4:
            return Gesture.STOP, 0.7
        
        return Gesture.UNKNOWN, 0.3
    
    def _distance(self, p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        """Calculate Euclidean distance between two points."""
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


# =============================================================================
# Face Manager
# =============================================================================

class FaceManager:
    """
    Manages face detection, tracking, and recognition.
    
    Uses MediaPipe for detection and optionally face_recognition for encoding.
    """
    
    def __init__(self, faces_dir: str = "data/faces", logger=None):
        self.logger = logger or get_logger(__name__)
        self.faces_dir = Path(faces_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
        # Known face encodings
        self.known_encodings: Dict[str, np.ndarray] = {}
        self.known_names: List[str] = []
        
        # Currently tracked faces
        self.tracked_faces: Dict[str, DetectedFace] = {}
        self._face_counter = 0
        
        # Load known faces
        self._load_known_faces()
    
    def _load_known_faces(self) -> None:
        """Load known face encodings from disk."""
        encodings_file = self.faces_dir / "encodings.pkl"
        
        if encodings_file.exists():
            try:
                with open(encodings_file, "rb") as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get("encodings", {})
                    self.known_names = list(self.known_encodings.keys())
                    self.logger.info(f"Loaded {len(self.known_names)} known faces")
            except Exception as e:
                self.logger.error(f"Failed to load face encodings: {e}")
    
    def _save_known_faces(self) -> None:
        """Save known face encodings to disk."""
        encodings_file = self.faces_dir / "encodings.pkl"
        
        try:
            with open(encodings_file, "wb") as f:
                pickle.dump({"encodings": self.known_encodings}, f)
            self.logger.info(f"Saved {len(self.known_encodings)} face encodings")
        except Exception as e:
            self.logger.error(f"Failed to save face encodings: {e}")
    
    def add_face(self, name: str, encoding: np.ndarray) -> bool:
        """
        Add a new face to the known faces database.
        
        Args:
            name: Name for this face
            encoding: Face encoding vector
        
        Returns:
            True if successful
        """
        self.known_encodings[name] = encoding
        if name not in self.known_names:
            self.known_names.append(name)
        self._save_known_faces()
        self.logger.info(f"Added face for: {name}")
        return True
    
    def recognize(self, encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Try to recognize a face from its encoding.
        
        Args:
            encoding: Face encoding to match
        
        Returns:
            Tuple of (name, confidence) or (None, 0.0)
        """
        if not FACE_RECOGNITION_AVAILABLE or not self.known_encodings:
            return None, 0.0
        
        # Compare to all known faces
        known_list = list(self.known_encodings.values())
        names_list = list(self.known_encodings.keys())
        
        if not known_list:
            return None, 0.0
        
        # Calculate distances
        distances = face_recognition.face_distance(known_list, encoding)
        
        if len(distances) == 0:
            return None, 0.0
        
        # Find best match
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0.0, 1.0 - min_distance)
        
        if confidence > 0.5:  # Recognition threshold
            return names_list[min_idx], confidence
        
        return None, confidence
    
    def generate_face_id(self) -> str:
        """Generate a unique face ID for tracking."""
        self._face_counter += 1
        return f"face_{self._face_counter}_{int(time.time())}"
    
    def update_tracking(self, current_faces: List[DetectedFace]) -> Tuple[List[str], List[str]]:
        """
        Update face tracking state.
        
        Args:
            current_faces: List of currently detected faces
        
        Returns:
            Tuple of (new_face_ids, lost_face_ids)
        """
        now = datetime.utcnow()
        current_ids = {f.face_id for f in current_faces}
        tracked_ids = set(self.tracked_faces.keys())
        
        # New faces
        new_ids = current_ids - tracked_ids
        
        # Lost faces (not seen for > 2 seconds)
        lost_ids = []
        for face_id, face in list(self.tracked_faces.items()):
            if face_id not in current_ids:
                time_since_seen = (now - face.last_seen).total_seconds()
                if time_since_seen > 2.0:
                    lost_ids.append(face_id)
                    del self.tracked_faces[face_id]
        
        # Update tracked faces
        for face in current_faces:
            if face.face_id in self.tracked_faces:
                self.tracked_faces[face.face_id].last_seen = now
                self.tracked_faces[face.face_id].bounding_box = face.bounding_box
            else:
                self.tracked_faces[face.face_id] = face
        
        return list(new_ids), lost_ids
    
    def has_faces(self) -> bool:
        """Check if there are any known faces enrolled."""
        return len(self.known_encodings) > 0
    
    def identify_face(self, encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify a face from its encoding.
        
        This is an alias for recognize() for clearer API.
        
        Args:
            encoding: Face encoding to match
        
        Returns:
            Tuple of (name, confidence) or (None, 0.0)
        """
        return self.recognize(encoding)


# =============================================================================
# Vision Processing
# =============================================================================

def _convert_to_hand_landmarks(mp_landmarks) -> HandLandmarks:
    """Convert MediaPipe NormalizedLandmarkList to our HandLandmarks dataclass."""
    def get_point(lm) -> Tuple[float, float, float]:
        return (lm.x, lm.y, lm.z)
    
    return HandLandmarks(
        wrist=get_point(mp_landmarks.landmark[0]),
        thumb_cmc=get_point(mp_landmarks.landmark[1]),
        thumb_mcp=get_point(mp_landmarks.landmark[2]),
        thumb_ip=get_point(mp_landmarks.landmark[3]),
        thumb_tip=get_point(mp_landmarks.landmark[4]),
        index_mcp=get_point(mp_landmarks.landmark[5]),
        index_pip=get_point(mp_landmarks.landmark[6]),
        index_dip=get_point(mp_landmarks.landmark[7]),
        index_tip=get_point(mp_landmarks.landmark[8]),
        middle_mcp=get_point(mp_landmarks.landmark[9]),
        middle_pip=get_point(mp_landmarks.landmark[10]),
        middle_dip=get_point(mp_landmarks.landmark[11]),
        middle_tip=get_point(mp_landmarks.landmark[12]),
        ring_mcp=get_point(mp_landmarks.landmark[13]),
        ring_pip=get_point(mp_landmarks.landmark[14]),
        ring_dip=get_point(mp_landmarks.landmark[15]),
        ring_tip=get_point(mp_landmarks.landmark[16]),
        pinky_mcp=get_point(mp_landmarks.landmark[17]),
        pinky_pip=get_point(mp_landmarks.landmark[18]),
        pinky_dip=get_point(mp_landmarks.landmark[19]),
        pinky_tip=get_point(mp_landmarks.landmark[20]),
    )


def _vision_process_main(
    camera_id: int,
    running_flag: Value,
    event_queue: Queue,
    command_queue: Queue,
    enable_gestures: bool = True,
    enable_faces: bool = True,
    show_preview: bool = True,
):
    """
    Main vision processing loop that runs in a separate process.
    This is necessary on macOS where OpenCV GUI must run in the main thread of a process.
    
    Args:
        camera_id: Camera device ID
        running_flag: Shared Value to control loop
        event_queue: Queue to send events to main process
        command_queue: Queue to receive commands from main process
        enable_gestures: Enable gesture detection
        enable_faces: Enable face detection
        show_preview: Show preview window
    """
    import logging
    logger = logging.getLogger("VisionProcess")
    
    # Face manager for enrollment
    face_manager = FaceManager(logger=logger)
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        event_queue.put({"type": "error", "message": f"Failed to open camera {camera_id}"})
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection
    
    hands = None
    face_detection = None
    
    if enable_gestures:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
    
    if enable_faces:
        face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
    
    # State
    gesture_recognizer = GestureRecognizer()
    last_gesture = None
    gesture_cooldown = 0.0
    presence_state = False
    frame_count = 0
    
    event_queue.put({"type": "started"})
    
    # Create window
    if show_preview:
        cv2.namedWindow("JARVIS Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("JARVIS Vision", 640, 480)
    
    while running_flag.value:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process gestures
            if enable_gestures and hands and frame_count % 2 == 0:
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                        )
                        
                        # Convert to our HandLandmarks format and recognize gesture
                        converted_landmarks = _convert_to_hand_landmarks(hand_landmarks)
                        gesture, confidence = gesture_recognizer.recognize(converted_landmarks)
                        current_time = time.time()
                        
                        if gesture != Gesture.UNKNOWN and confidence >= 0.7:
                            if gesture != last_gesture or current_time > gesture_cooldown:
                                last_gesture = gesture
                                gesture_cooldown = current_time + 1.0  # 1 second cooldown
                                
                                # Send gesture event
                                event_queue.put({
                                    "type": "gesture",
                                    "gesture": gesture.value,
                                    "confidence": confidence,
                                })
            
            # Process faces
            if enable_faces and face_detection and frame_count % 3 == 0:
                results = face_detection.process(rgb_frame)
                faces_detected = False
                
                if results.detections:
                    faces_detected = True
                    h, w = display_frame.shape[:2]
                    
                    # Try to recognize faces if face_recognition is available
                    face_names = []
                    if FACE_RECOGNITION_AVAILABLE and face_manager.has_faces():
                        try:
                            # Find face locations and encodings
                            face_locations = face_recognition.face_locations(rgb_frame)
                            if face_locations:
                                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                                
                                for encoding in face_encodings:
                                    # Try to identify the face
                                    name, confidence = face_manager.identify_face(encoding)
                                    if name:
                                        face_names.append(name)
                                    else:
                                        face_names.append("Unknown")
                        except Exception as e:
                            logger.warning(f"Face recognition error: {e}")
                    
                    for idx, detection in enumerate(results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        bw = int(bbox.width * w)
                        bh = int(bbox.height * h)
                        
                        # Get name for this face (if recognized)
                        face_label = "Face"
                        if idx < len(face_names):
                            face_label = face_names[idx]
                        
                        # Draw bounding box - use different color for recognized faces
                        if face_label != "Face" and face_label != "Unknown":
                            color = (0, 255, 255)  # Yellow for recognized
                        else:
                            color = (0, 255, 0)  # Green for unknown
                        
                        cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), color, 2)
                        cv2.putText(display_frame, face_label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Check presence change
                if faces_detected != presence_state:
                    presence_state = faces_detected
                    event_queue.put({
                        "type": "presence",
                        "present": presence_state,
                    })
            
            # Draw overlay
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, "JARVIS VISION", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if last_gesture:
                cv2.putText(display_frame, f"Gesture: {last_gesture.value}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            status = "User Present" if presence_state else "No User"
            color = (0, 255, 0) if presence_state else (0, 0, 255)
            cv2.putText(display_frame, status, (w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(display_frame, "Press 'Q' to close", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Check for commands from main process
            try:
                while not command_queue.empty():
                    cmd = command_queue.get_nowait()
                    if cmd.get("command") == "enroll_face":
                        name = cmd.get("name", "unknown")
                        # Capture current frame and enroll
                        if FACE_RECOGNITION_AVAILABLE:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb)
                            if face_locations:
                                encodings = face_recognition.face_encodings(rgb, face_locations)
                                if encodings:
                                    success = face_manager.add_face(name, encodings[0])
                                    event_queue.put({
                                        "type": "enroll_result",
                                        "success": success,
                                        "name": name,
                                    })
                                    logger.info(f"Face enrolled for {name}: {success}")
                                else:
                                    event_queue.put({
                                        "type": "enroll_result",
                                        "success": False,
                                        "name": name,
                                        "error": "Failed to encode face",
                                    })
                            else:
                                event_queue.put({
                                    "type": "enroll_result",
                                    "success": False,
                                    "name": name,
                                    "error": "No face detected - please look at the camera",
                                })
                        else:
                            event_queue.put({
                                "type": "enroll_result",
                                "success": False,
                                "name": name,
                                "error": "face_recognition library not available",
                            })
            except Exception as cmd_error:
                logger.error(f"Command error: {cmd_error}")
            
            # Show preview
            if show_preview:
                cv2.imshow("JARVIS Vision", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    running_flag.value = 0
                    break
            
        except Exception as e:
            logger.error(f"Vision process error: {e}")
            time.sleep(0.1)
    
    # Cleanup
    if hands:
        hands.close()
    if face_detection:
        face_detection.close()
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    event_queue.put({"type": "stopped"})


class VisionProcessor:
    """
    Main vision processing class.
    
    Handles camera capture, gesture detection, and face detection.
    Uses multiprocessing on macOS (required for OpenCV GUI) and threading elsewhere.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        enable_gestures: bool = True,
        enable_faces: bool = True,
        show_preview: bool = True,
        logger=None,
    ):
        self.logger = logger or get_logger(__name__)
        self.camera_id = camera_id
        self.enable_gestures = enable_gestures
        self.enable_faces = enable_faces
        self.show_preview = show_preview
        
        # Detect if we're on macOS
        self._use_multiprocessing = platform.system() == "Darwin" and show_preview
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[Process] = None
        self._running_flag: Optional[Value] = None  # For multiprocessing
        self._event_queue: Optional[Queue] = None  # For multiprocessing events
        self._command_queue: Optional[Queue] = None  # For multiprocessing commands
        self._event_handler_thread: Optional[threading.Thread] = None
        self._cap: Optional[Any] = None  # cv2.VideoCapture
        
        # Frame for display (shared between threads)
        self._display_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        
        # MediaPipe components (for threading mode)
        self._mp_hands = None
        self._mp_face_detection = None
        self._hands = None
        self._face_detection = None
        
        # Recognizers
        self.gesture_recognizer = GestureRecognizer(logger=self.logger)
        self.face_manager = FaceManager(logger=self.logger)
        
        # Callbacks for event emission
        self.on_gesture: Optional[Callable[[Gesture, float, str, Dict], None]] = None
        self.on_face_detected: Optional[Callable[[DetectedFace], None]] = None
        self.on_face_lost: Optional[Callable[[str, float], None]] = None
        self.on_presence_changed: Optional[Callable[[bool, int], None]] = None
        self.on_enrollment_complete: Optional[Callable[[bool, str, str], None]] = None
        
        # State tracking
        self._last_gesture: Optional[Gesture] = None
        self._gesture_cooldown = 0.0
        self._presence_state = False
        self._frame_count = 0
        
        # Performance
        self._skip_frames = 2  # Process every Nth frame for performance
    
    def start(self) -> bool:
        """Start the vision processing."""
        if not CV2_AVAILABLE:
            self.logger.error("OpenCV not available - install with: pip install opencv-python")
            return False
        
        if not MEDIAPIPE_AVAILABLE:
            self.logger.error("MediaPipe not available - install with: pip install mediapipe")
            return False
        
        if self._running:
            self.logger.warning("Vision processor already running")
            return True
        
        try:
            self.logger.info(f"Vision start: platform={platform.system()}, show_preview={self.show_preview}, use_multiprocessing={self._use_multiprocessing}")
            
            if self._use_multiprocessing:
                # macOS: Use separate process for OpenCV GUI
                return self._start_multiprocess()
            else:
                # Linux/Windows or no preview: Use threading
                return self._start_threaded()
            
        except Exception as e:
            self.logger.error(f"Failed to start vision processor: {e}")
            self._cleanup()
            return False
    
    def _start_multiprocess(self) -> bool:
        """Start vision processing in a separate process (for macOS)."""
        self.logger.info("Starting vision in multiprocess mode (macOS)")
        
        # Create shared state
        self._running_flag = Value('i', 1)  # 1 = running
        self._event_queue = Queue()
        self._command_queue = Queue()  # For sending commands to the process
        
        # Start the vision process
        self._process = Process(
            target=_vision_process_main,
            args=(
                self.camera_id,
                self._running_flag,
                self._event_queue,
                self._command_queue,  # Pass command queue (4th position)
                self.enable_gestures,
                self.enable_faces,
                self.show_preview,
            ),
            daemon=True,
        )
        self._process.start()
        
        # Start event handler thread to process events from the vision process
        self._running = True
        self._event_handler_thread = threading.Thread(target=self._handle_process_events, daemon=True)
        self._event_handler_thread.start()
        
        self.logger.info("Vision processor started (multiprocess mode)")
        return True
    
    def _start_threaded(self) -> bool:
        """Start vision processing in a thread (for Linux/Windows)."""
        # Initialize camera
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties for performance
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize MediaPipe
        self._init_mediapipe()
        
        # Start processing thread
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
        self.logger.info("Vision processor started (threaded mode)")
        return True
    
    def _handle_process_events(self) -> None:
        """Handle events from the vision process."""
        while self._running:
            try:
                if self._event_queue and not self._event_queue.empty():
                    event = self._event_queue.get(timeout=0.1)
                    
                    if event["type"] == "gesture":
                        gesture = Gesture(event["gesture"])
                        confidence = event["confidence"]
                        self._last_gesture = gesture
                        
                        if self.on_gesture:
                            self.on_gesture(gesture, confidence, "right", {})
                    
                    elif event["type"] == "presence":
                        self._presence_state = event["present"]
                        if self.on_presence_changed:
                            self.on_presence_changed(event["present"], 1 if event["present"] else 0)
                    
                    elif event["type"] == "enroll_result":
                        # Handle face enrollment result
                        success = event.get("success", False)
                        name = event.get("name", "unknown")
                        message = event.get("error", "")
                        self.logger.info(f"Enrollment result for {name}: success={success}, message={message}")
                        
                        # Trigger a callback if available
                        if hasattr(self, 'on_enrollment_complete') and self.on_enrollment_complete:
                            self.on_enrollment_complete(success, name, message)
                    
                    elif event["type"] == "error":
                        self.logger.error(f"Vision process error: {event['message']}")
                    
                    elif event["type"] == "stopped":
                        self._running = False
                        break
                        
                else:
                    time.sleep(0.01)
                    
            except Exception:
                time.sleep(0.01)
    
    def stop(self) -> None:
        """Stop the vision processing."""
        self._running = False
        
        if self._use_multiprocessing:
            # Stop the process
            if self._running_flag:
                self._running_flag.value = 0
            
            if self._process:
                self._process.join(timeout=3.0)
                if self._process.is_alive():
                    self._process.terminate()
                self._process = None
            
            if self._event_handler_thread:
                self._event_handler_thread.join(timeout=1.0)
                self._event_handler_thread = None
        else:
            # Close display window
            if self.show_preview and CV2_AVAILABLE:
                try:
                    cv2.destroyWindow("JARVIS Vision")
                    cv2.waitKey(1)
                except Exception:
                    pass
            
            if self._thread:
                self._thread.join(timeout=2.0)
                self._thread = None
            
            self._cleanup()
        
        self.logger.info("Vision processor stopped")
    
    def enroll_face(self, name: str) -> None:
        """Request face enrollment in the vision process.
        
        Args:
            name: The name to associate with the face
        """
        if self._use_multiprocessing and self._command_queue:
            self.logger.info(f"Sending enroll_face command for: {name}")
            self._command_queue.put({
                "command": "enroll_face",
                "name": name
            })
        else:
            # For threaded mode, we'd handle enrollment differently
            self.logger.warning("Face enrollment only supported in multiprocess mode")
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe components."""
        self._mp_hands = mp.solutions.hands
        self._mp_face_detection = mp.solutions.face_detection
        
        if self.enable_gestures:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        
        if self.enable_faces:
            self._face_detection = self._mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5,
            )
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if self._hands:
            self._hands.close()
            self._hands = None
        
        if self._face_detection:
            self._face_detection.close()
            self._face_detection = None
    
    def _process_loop(self) -> None:
        """Main processing loop running in background thread.
        
        NOTE: This method should NOT be called on macOS with show_preview=True
        because OpenCV GUI doesn't work in threads on macOS.
        Use multiprocessing mode instead.
        """
        # On macOS, skip the preview window in threaded mode (it will crash)
        is_macos = platform.system() == "Darwin"
        use_preview = self.show_preview and not is_macos
        
        # Create window for preview (if enabled and NOT on macOS)
        if use_preview:
            cv2.namedWindow("JARVIS Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("JARVIS Vision", 640, 480)
        
        while self._running:
            try:
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                self._frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Create display frame for annotations
                display_frame = frame.copy()
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process gestures
                if self.enable_gestures and self._hands:
                    self._process_hands(rgb_frame, display_frame)
                
                # Process faces
                if self.enable_faces and self._face_detection:
                    self._process_faces(rgb_frame, display_frame)
                
                # Draw status overlay
                self._draw_status_overlay(display_frame)
                
                # Show preview window (only on non-macOS in threaded mode)
                if use_preview:
                    cv2.imshow("JARVIS Vision", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    # Press 'q' to quit, ESC to close
                    if key == ord('q') or key == 27:
                        self._running = False
                        break
                
                # Store frame for external access
                with self._frame_lock:
                    self._display_frame = display_frame.copy()
                
            except Exception as e:
                self.logger.error(f"Error in vision processing loop: {e}")
                time.sleep(0.1)
        
        # Cleanup window
        if use_preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    
    def _draw_status_overlay(self, frame: np.ndarray) -> None:
        """Draw status information on the frame."""
        h, w = frame.shape[:2]
        
        # Draw "JARVIS VISION" title
        cv2.putText(
            frame, "JARVIS VISION", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Draw gesture if detected
        if self._last_gesture and self._last_gesture != Gesture.UNKNOWN:
            cv2.putText(
                frame, f"Gesture: {self._last_gesture.value}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
        
        # Draw presence indicator
        status = "User Present" if self._presence_state else "No User"
        color = (0, 255, 0) if self._presence_state else (0, 0, 255)
        cv2.putText(
            frame, status, (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        
        # Draw instructions
        cv2.putText(
            frame, "Press 'Q' to close", (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1
        )
    
    def _process_hands(self, rgb_frame: np.ndarray, display_frame: np.ndarray) -> None:
        """Process hand detection and gesture recognition."""
        results = self._hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Draw hand landmarks on display frame
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get handedness
                handedness = "Right"
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                
                # Convert landmarks to our format
                landmarks = self._convert_landmarks(hand_landmarks)
                
                # Recognize gesture
                gesture, confidence = self.gesture_recognizer.recognize(landmarks, handedness)
                
                # Check cooldown and emit event
                current_time = time.time()
                if (
                    gesture != Gesture.UNKNOWN
                    and confidence >= 0.7
                    and (gesture != self._last_gesture or current_time > self._gesture_cooldown)
                ):
                    self._last_gesture = gesture
                    self._gesture_cooldown = current_time + 1.0  # 1 second cooldown
                    
                    if self.on_gesture:
                        # Calculate bounding box
                        h, w = rgb_frame.shape[:2]
                        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                        bbox = {
                            "x": min(x_coords) / w,
                            "y": min(y_coords) / h,
                            "width": (max(x_coords) - min(x_coords)) / w,
                            "height": (max(y_coords) - min(y_coords)) / h,
                        }
                        
                        self.on_gesture(gesture, confidence, handedness.lower(), bbox)
    
    def _convert_landmarks(self, hand_landmarks) -> HandLandmarks:
        """Convert MediaPipe landmarks to our format."""
        lm = hand_landmarks.landmark
        
        def get_point(idx: int) -> Tuple[float, float, float]:
            return (lm[idx].x, lm[idx].y, lm[idx].z)
        
        return HandLandmarks(
            wrist=get_point(0),
            thumb_cmc=get_point(1),
            thumb_mcp=get_point(2),
            thumb_ip=get_point(3),
            thumb_tip=get_point(4),
            index_mcp=get_point(5),
            index_pip=get_point(6),
            index_dip=get_point(7),
            index_tip=get_point(8),
            middle_mcp=get_point(9),
            middle_pip=get_point(10),
            middle_dip=get_point(11),
            middle_tip=get_point(12),
            ring_mcp=get_point(13),
            ring_pip=get_point(14),
            ring_dip=get_point(15),
            ring_tip=get_point(16),
            pinky_mcp=get_point(17),
            pinky_pip=get_point(18),
            pinky_dip=get_point(19),
            pinky_tip=get_point(20),
        )
    
    def _process_faces(self, rgb_frame: np.ndarray, display_frame: np.ndarray) -> None:
        """Process face detection and recognition."""
        results = self._face_detection.process(rgb_frame)
        
        detected_faces: List[DetectedFace] = []
        h, w = rgb_frame.shape[:2]
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure bounds are valid
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Draw face bounding box on display frame
                cv2.rectangle(display_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Generate face ID (simple tracking by position)
                face_id = self._get_or_create_face_id(x, y, width, height)
                
                # Get face encoding if face_recognition is available
                encoding = None
                name = None
                confidence = detection.score[0] if detection.score else 0.0
                
                if FACE_RECOGNITION_AVAILABLE and self._frame_count % 10 == 0:
                    # Only encode every 10th frame for performance
                    try:
                        face_locations = [(y, x + width, y + height, x)]
                        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        if encodings:
                            encoding = encodings[0]
                            name, rec_confidence = self.face_manager.recognize(encoding)
                            if name:
                                confidence = rec_confidence
                    except Exception as e:
                        self.logger.debug(f"Face encoding failed: {e}")
                
                # Draw name label if recognized
                label = name if name else f"ID: {face_id[:8]}"
                cv2.putText(
                    display_frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                
                face = DetectedFace(
                    face_id=face_id,
                    bounding_box=(x, y, width, height),
                    landmarks={},
                    encoding=encoding,
                    name=name,
                    confidence=confidence,
                )
                detected_faces.append(face)
        
        # Update tracking
        new_ids, lost_ids = self.face_manager.update_tracking(detected_faces)
        
        # Emit events for new faces
        for face in detected_faces:
            if face.face_id in new_ids and self.on_face_detected:
                self.on_face_detected(face)
        
        # Emit events for lost faces
        for face_id in lost_ids:
            if self.on_face_lost:
                self.on_face_lost(face_id, 0.0)
        
        # Check presence change
        current_presence = len(detected_faces) > 0
        if current_presence != self._presence_state:
            self._presence_state = current_presence
            if self.on_presence_changed:
                self.on_presence_changed(current_presence, len(detected_faces))
    
    def _get_or_create_face_id(self, x: int, y: int, w: int, h: int) -> str:
        """Get existing face ID or create new one based on position."""
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check if this matches an existing tracked face
        for face_id, face in self.face_manager.tracked_faces.items():
            fx, fy, fw, fh = face.bounding_box
            fcx = fx + fw // 2
            fcy = fy + fh // 2
            
            # If centers are close, it's the same face
            if abs(center_x - fcx) < w and abs(center_y - fcy) < h:
                return face_id
        
        return self.face_manager.generate_face_id()
    
    def learn_face(self, name: str, frame: Optional[np.ndarray] = None) -> bool:
        """
        Learn a new face from current camera frame.
        
        Args:
            name: Name to associate with this face
            frame: Optional frame to use (captures new one if not provided)
        
        Returns:
            True if face was learned successfully
        """
        if not FACE_RECOGNITION_AVAILABLE:
            self.logger.error("face_recognition library not available")
            return False
        
        if frame is None and self._cap:
            ret, frame = self._cap.read()
            if not ret:
                self.logger.error("Failed to capture frame")
                return False
        
        if frame is None:
            return False
        
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect and encode face
        face_locations = face_recognition.face_locations(rgb)
        if not face_locations:
            self.logger.warning("No face detected in frame")
            return False
        
        encodings = face_recognition.face_encodings(rgb, face_locations)
        if not encodings:
            self.logger.warning("Failed to encode face")
            return False
        
        # Save the encoding
        return self.face_manager.add_face(name, encodings[0])


# =============================================================================
# Vision Agent
# =============================================================================

class VisionAgent(BaseAgent):
    """
    Vision Agent for gesture detection and face recognition.
    
    Integrates OpenCV and MediaPipe to provide visual input capabilities.
    All communication is via EventBus - no direct agent calls.
    
    Events Subscribed:
        - VisionStartEvent: Start vision processing
        - VisionStopEvent: Stop vision processing
        - VisionCommandEvent: Execute vision commands
        - ShutdownRequestedEvent: Clean shutdown
    
    Events Emitted:
        - GestureDetectedEvent: When a gesture is recognized
        - FaceDetectedEvent: When a face is detected/recognized
        - FaceLostEvent: When a tracked face is lost
        - PresenceChangedEvent: When user presence changes
        - VisionCommandResultEvent: Result of vision commands
    """
    
    def __init__(self, config: Any = None):
        super().__init__(name="VisionAgent", config=config)
        
        # Configuration
        self._vision_config = {}
        if config and isinstance(config, dict):
            self._vision_config = config.get("vision", {})
        
        # Camera settings (nested under vision.camera)
        camera_config = self._vision_config.get("camera", {})
        self.camera_id = camera_config.get("device_id", 0)
        self.auto_start = camera_config.get("auto_start", False)
        self.show_preview = camera_config.get("show_preview", True)
        
        # Gesture settings (nested under vision.gestures)
        gestures_config = self._vision_config.get("gestures", {})
        self.enable_gestures = gestures_config.get("enabled", True)
        
        # Face settings (nested under vision.face)
        face_config = self._vision_config.get("face", {})
        self.enable_faces = face_config.get("enabled", True)
        
        # Vision processor
        self.processor: Optional[VisionProcessor] = None
        
        # Agent logger for structured logging
        self._agent_logger = get_agent_logger("VisionAgent")
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent provides."""
        caps = []
        
        if self.enable_gestures:
            caps.append(AgentCapability(
                name="gesture_detection",
                description="Detect and recognize hand gestures",
                input_events=["VisionStartEvent", "VisionCommandEvent"],
                output_events=["GestureDetectedEvent"],
            ))
        
        if self.enable_faces:
            caps.append(AgentCapability(
                name="face_detection",
                description="Detect and track faces",
                input_events=["VisionStartEvent", "VisionCommandEvent"],
                output_events=["FaceDetectedEvent", "FaceLostEvent"],
            ))
            caps.append(AgentCapability(
                name="face_recognition",
                description="Recognize known faces",
                input_events=["VisionCommandEvent"],
                output_events=["FaceDetectedEvent", "VisionCommandResultEvent"],
            ))
            caps.append(AgentCapability(
                name="presence_detection",
                description="Detect user presence via camera",
                input_events=["VisionStartEvent"],
                output_events=["PresenceChangedEvent"],
            ))
        
        return caps
    
    async def _setup(self) -> None:
        """Initialize the agent."""
        # Check dependencies
        if not CV2_AVAILABLE:
            self._logger.warning("OpenCV not available - vision features disabled")
            self._logger.warning("Install with: pip install opencv-python")
        
        if not MEDIAPIPE_AVAILABLE:
            self._logger.warning("MediaPipe not available - vision features disabled")
            self._logger.warning("Install with: pip install mediapipe")
        
        if not FACE_RECOGNITION_AVAILABLE:
            self._logger.info("face_recognition not available - using basic face detection only")
            self._logger.info("For recognition, install: pip install face_recognition")
        
        # Subscribe to events
        self._subscribe(ShutdownRequestedEvent, self._handle_shutdown)
        self._subscribe(VisionStartEvent, self._handle_start)
        self._subscribe(VisionStopEvent, self._handle_stop)
        self._subscribe(VisionCommandEvent, self._handle_command)
        self._subscribe(IntentRecognizedEvent, self._handle_intent)
        
        # Initialize processor (but don't start camera yet)
        if CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
            self.processor = VisionProcessor(
                camera_id=self.camera_id,
                enable_gestures=self.enable_gestures,
                enable_faces=self.enable_faces,
                show_preview=self.show_preview,
                logger=self._logger,
            )
            
            # Set up callbacks
            self.processor.on_gesture = self._on_gesture
            self.processor.on_face_detected = self._on_face_detected
            self.processor.on_face_lost = self._on_face_lost
            self.processor.on_presence_changed = self._on_presence_changed
            self.processor.on_enrollment_complete = self._on_enrollment_complete
            
            # Auto-start if configured
            if self.auto_start:
                self.processor.start()
        
        self._logger.info("Vision agent initialized")
    
    async def _teardown(self) -> None:
        """Clean up the agent."""
        if self.processor:
            self.processor.stop()
        
        self._logger.info("Vision agent shutdown complete")
    
    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------
    
    async def _handle_shutdown(self, event: ShutdownRequestedEvent) -> None:
        """Handle shutdown request."""
        self._logger.info(f"Received shutdown request: {event.reason}")
        await self.stop(reason=event.reason)
    
    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        """Handle recognized intents for vision control."""
        intent = event.intent.upper()
        
        # Check if this intent is for VisionAgent
        if intent not in ("START_VISION", "STOP_VISION", "ENROLL_FACE", "TOGGLE_VISION"):
            return
        
        self._agent_logger.event_received("IntentRecognizedEvent", str(event.event_id), event.source)
        self._logger.info(f"Handling vision intent: {intent}")
        
        if intent in ("START_VISION", "TOGGLE_VISION"):
            if self.processor and self.processor._running:
                # Already running - for toggle, stop it
                if intent == "TOGGLE_VISION":
                    self.processor.stop()
                    self._logger.info("Vision processing stopped via toggle")
                    await self._emit(VoiceOutputEvent(
                        text="Vision system deactivated.",
                        source="VisionAgent",
                    ))
                else:
                    await self._emit(VoiceOutputEvent(
                        text="Vision system is already active.",
                        source="VisionAgent",
                    ))
            else:
                # Start vision
                if self.processor:
                    success = self.processor.start()
                    if success:
                        self._logger.info("Vision processing started")
                        await self._emit(VoiceOutputEvent(
                            text="Vision system activated. I can now see gestures and faces.",
                            source="VisionAgent",
                        ))
                    else:
                        self._logger.error("Failed to start vision processing")
                        await self._emit(VoiceOutputEvent(
                            text="Failed to start vision system. Camera may not be available.",
                            source="VisionAgent",
                        ))
                else:
                    self._logger.error("Vision processor not available")
                    await self._emit(VoiceOutputEvent(
                        text="Vision system is not available. Required libraries may not be installed.",
                        source="VisionAgent",
                    ))
        
        elif intent == "STOP_VISION":
            if self.processor and self.processor._running:
                self.processor.stop()
                self._logger.info("Vision processing stopped")
                await self._emit(VoiceOutputEvent(
                    text="Vision system deactivated.",
                    source="VisionAgent",
                ))
            else:
                await self._emit(VoiceOutputEvent(
                    text="Vision system is not currently active.",
                    source="VisionAgent",
                ))
        
        elif intent == "ENROLL_FACE":
            name = event.entities.get("name", "unknown")
            if self.processor and FACE_RECOGNITION_AVAILABLE:
                # Capture current face and enroll
                await self._emit(VoiceOutputEvent(
                    text=f"Please look at the camera. Enrolling face as {name}.",
                    source="VisionAgent",
                ))
                # Send enrollment command to the vision process
                self.processor.enroll_face(name)
                self._logger.info(f"Face enrollment requested for: {name}")
            else:
                await self._emit(VoiceOutputEvent(
                    text="Face enrollment requires the face recognition library to be installed.",
                    source="VisionAgent",
                ))
    
    async def _handle_start(self, event: VisionStartEvent) -> None:
        """Handle vision start event."""
        self._agent_logger.event_received("VisionStartEvent", str(event.event_id), event.source)
        
        if not self.processor:
            self._logger.error("Vision processor not available")
            return
        
        # Update configuration
        self.enable_gestures = event.enable_gestures
        self.enable_faces = event.enable_faces
        self.processor.enable_gestures = event.enable_gestures
        self.processor.enable_faces = event.enable_faces
        
        if event.camera_id != self.camera_id:
            self.camera_id = event.camera_id
            self.processor.camera_id = event.camera_id
            # Restart processor with new camera
            self.processor.stop()
        
        success = self.processor.start()
        
        if success:
            self._logger.info("Vision processing started")
            await self._emit(VoiceOutputEvent(
                text="Vision system activated.",
                source="VisionAgent",
            ))
        else:
            self._logger.error("Failed to start vision processing")
    
    async def _handle_stop(self, event: VisionStopEvent) -> None:
        """Handle vision stop event."""
        self._agent_logger.event_received("VisionStopEvent", str(event.event_id), event.source)
        
        if self.processor:
            self.processor.stop()
            self._logger.info(f"Vision processing stopped: {event.reason}")
    
    async def _handle_command(self, event: VisionCommandEvent) -> None:
        """Handle vision command event."""
        start_time = time.perf_counter()
        self._agent_logger.event_received("VisionCommandEvent", str(event.event_id), event.source)
        
        command = event.command
        params = event.parameters
        
        success = False
        result: Dict[str, Any] = {}
        error: Optional[str] = None
        
        try:
            if command == "start":
                if self.processor:
                    success = self.processor.start()
                else:
                    error = "Vision processor not available"
            
            elif command == "stop":
                if self.processor:
                    self.processor.stop()
                    success = True
                else:
                    error = "Vision processor not available"
            
            elif command == "toggle_gestures":
                if self.processor:
                    self.processor.enable_gestures = not self.processor.enable_gestures
                    result["enabled"] = self.processor.enable_gestures
                    success = True
                else:
                    error = "Vision processor not available"
            
            elif command == "toggle_faces":
                if self.processor:
                    self.processor.enable_faces = not self.processor.enable_faces
                    result["enabled"] = self.processor.enable_faces
                    success = True
                else:
                    error = "Vision processor not available"
            
            elif command == "learn_face":
                name = params.get("name")
                if not name:
                    error = "Name parameter required"
                elif self.processor:
                    success = self.processor.learn_face(name)
                    if success:
                        result["name"] = name
                        result["message"] = f"Learned face for {name}"
                    else:
                        error = "Failed to learn face - ensure face is visible"
                else:
                    error = "Vision processor not available"
            
            elif command == "list_known_faces":
                if self.processor:
                    result["faces"] = self.processor.face_manager.known_names
                    success = True
                else:
                    error = "Vision processor not available"
            
            elif command == "status":
                result["running"] = self.processor._running if self.processor else False
                result["gestures_enabled"] = self.enable_gestures
                result["faces_enabled"] = self.enable_faces
                result["opencv_available"] = CV2_AVAILABLE
                result["mediapipe_available"] = MEDIAPIPE_AVAILABLE
                result["face_recognition_available"] = FACE_RECOGNITION_AVAILABLE
                success = True
            
            else:
                error = f"Unknown command: {command}"
        
        except Exception as e:
            error = str(e)
            self._logger.exception(f"Error executing vision command: {command}")
        
        # Emit result
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        await self._emit(VisionCommandResultEvent(
            command=command,
            success=success,
            result=result,
            error=error,
            source="VisionAgent",
        ))
        
        self._agent_logger.event_handled(
            "VisionCommandEvent",
            str(event.event_id),
            duration_ms,
            success=success,
            error=error or "",
        )
    
    # -------------------------------------------------------------------------
    # Callbacks (called from processor thread)
    # -------------------------------------------------------------------------
    
    def _on_gesture(
        self, 
        gesture: Gesture, 
        confidence: float, 
        hand: str,
        bounding_box: Dict[str, float],
    ) -> None:
        """Called when a gesture is detected."""
        self._logger.info(f"Gesture detected: {gesture.value} ({confidence:.2f}) - {hand} hand")
        
        # Schedule async event emission
        asyncio.run_coroutine_threadsafe(
            self.emit(GestureDetectedEvent(
                gesture=gesture.value,
                confidence=confidence,
                hand=hand,
                bounding_box=bounding_box,
                source="VisionAgent",
            )),
            asyncio.get_event_loop(),
        )
        
        # If gesture has an associated action, emit voice feedback
        if gesture in GESTURE_ACTIONS:
            action = GESTURE_ACTIONS[gesture]
            asyncio.run_coroutine_threadsafe(
                self.emit(VoiceOutputEvent(
                    text=f"Gesture recognized: {gesture.value.replace('_', ' ')}",
                    source="VisionAgent",
                )),
                asyncio.get_event_loop(),
            )
    
    def _on_face_detected(self, face: DetectedFace) -> None:
        """Called when a face is detected."""
        self._logger.debug(f"Face detected: {face.face_id}, recognized: {face.name}")
        
        asyncio.run_coroutine_threadsafe(
            self.emit(FaceDetectedEvent(
                face_id=face.face_id,
                is_recognized=face.name is not None,
                person_name=face.name or "",
                confidence=face.confidence,
                bounding_box={
                    "x": face.bounding_box[0],
                    "y": face.bounding_box[1],
                    "width": face.bounding_box[2],
                    "height": face.bounding_box[3],
                },
                landmarks=face.landmarks,
                source="VisionAgent",
            )),
            asyncio.get_event_loop(),
        )
        
        # Announce recognized person
        if face.name:
            asyncio.run_coroutine_threadsafe(
                self.emit(VoiceOutputEvent(
                    text=f"Hello {face.name}",
                    source="VisionAgent",
                )),
                asyncio.get_event_loop(),
            )
    
    def _on_face_lost(self, face_id: str, duration: float) -> None:
        """Called when a face is lost."""
        self._logger.debug(f"Face lost: {face_id}")
        
        asyncio.run_coroutine_threadsafe(
            self.emit(FaceLostEvent(
                face_id=face_id,
                duration_seconds=duration,
                source="VisionAgent",
            )),
            asyncio.get_event_loop(),
        )
    
    def _on_presence_changed(self, is_present: bool, face_count: int) -> None:
        """Called when user presence changes."""
        self._logger.info(f"Presence changed: present={is_present}, faces={face_count}")
        
        asyncio.run_coroutine_threadsafe(
            self.emit(PresenceChangedEvent(
                is_present=is_present,
                face_count=face_count,
                source="VisionAgent",
            )),
            asyncio.get_event_loop(),
        )

    def _on_enrollment_complete(self, success: bool, name: str, message: str) -> None:
        """Called when face enrollment completes."""
        self._logger.info(f"Enrollment complete for {name}: success={success}, {message}")
        
        # Emit a voice response about the result
        if success:
            text = f"Face enrolled successfully as {name}."
        else:
            text = f"Failed to enroll face: {message}"
        
        asyncio.run_coroutine_threadsafe(
            self.emit(VoiceOutputEvent(
                text=text,
                source="VisionAgent",
            )),
            asyncio.get_event_loop(),
        )
