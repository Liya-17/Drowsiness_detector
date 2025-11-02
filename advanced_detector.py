"""
Advanced Drowsiness Detection System
Uses MediaPipe Face Mesh for high-precision facial landmark detection
Implements multiple advanced fatigue detection algorithms
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import time
from datetime import datetime
import json


class AdvancedDrowsinessDetector:
    """
    Advanced drowsiness detection using MediaPipe Face Mesh (478 landmarks)
    with multi-modal fatigue analysis and adaptive thresholding
    """

    # MediaPipe Face Mesh landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    # Separate upper and lower lip for better visualization
    UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_LEFT = 33
    RIGHT_EYE_RIGHT = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    def __init__(self, config):
        """Initialize the advanced detector with configuration"""
        self.config = config

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Thresholds (will be adapted during calibration)
        self.ear_threshold = config['thresholds']['ear_baseline']
        self.mar_threshold = config['thresholds']['mar_baseline']
        self.perclos_warning = config['thresholds']['perclos_warning']
        self.perclos_critical = config['thresholds']['perclos_critical']
        self.microsleep_frames = config['thresholds']['microsleep_duration']

        # Calibration data
        self.is_calibrated = False
        self.baseline_ear = None
        self.baseline_mar = None
        self.baseline_pupil_size = None
        self.calibration_data = {
            'ear_values': [],
            'mar_values': [],
            'pupil_sizes': []
        }

        # Time series buffers for temporal analysis
        self.ear_history = deque(maxlen=300)  # 10 seconds at 30fps
        self.mar_history = deque(maxlen=300)
        self.eye_closure_history = deque(maxlen=300)
        self.blink_timestamps = deque(maxlen=100)
        self.yawn_timestamps = deque(maxlen=50)
        self.head_pose_history = deque(maxlen=100)
        self.pupil_size_history = deque(maxlen=300)
        self.gaze_direction_history = deque(maxlen=150)

        # State tracking
        self.consecutive_closed_frames = 0
        self.consecutive_drowsy_frames = 0
        self.consecutive_mouth_open_frames = 0
        self.mouth_open_start_mar = 0
        self.microsleep_count = 0
        self.blink_count = 0
        self.yawn_count = 0
        self.total_frames = 0
        self.start_time = time.time()
        self.last_yawn_detected = False

        # Alert state
        self.alert_level = 0
        self.last_alert_time = {1: 0, 2: 0, 3: 0, 4: 0}
        self.alert_history = deque(maxlen=60)  # 2 seconds at 30fps

        # Advanced features
        self.features_enabled = config.get('features', {})

        # Session data
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'events': [],
            'metrics_snapshots': []
        }

    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # Vertical eye landmarks
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Horizontal eye landmark
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio
        MAR = (vertical_dist_1 + vertical_dist_2 + vertical_dist_3) / (3 * horizontal_dist)
        """
        # Simplified MAR calculation
        vertical_distances = []
        for i in range(3):
            idx_top = i * 4 + 2
            idx_bottom = i * 4 + 6
            if idx_top < len(mouth_landmarks) and idx_bottom < len(mouth_landmarks):
                vertical_distances.append(
                    dist.euclidean(mouth_landmarks[idx_top], mouth_landmarks[idx_bottom])
                )

        if len(vertical_distances) == 0:
            return 0.0

        horizontal_dist = dist.euclidean(mouth_landmarks[0], mouth_landmarks[-1])

        if horizontal_dist == 0:
            return 0.0

        mar = sum(vertical_distances) / (len(vertical_distances) * horizontal_dist)
        return mar

    def calculate_mouth_metrics(self, landmarks):
        """
        Calculate detailed mouth metrics for yawn detection
        Returns: (mar, vertical_opening, mouth_area, aspect_ratio)
        """
        # Get upper and lower lip landmarks
        upper_lip = landmarks[self.UPPER_LIP_INDICES]
        lower_lip = landmarks[self.LOWER_LIP_INDICES]

        # Calculate vertical opening (center of mouth)
        # Use points from center of upper and lower lips
        upper_center_idx = len(upper_lip) // 2
        lower_center_idx = len(lower_lip) // 2
        vertical_opening = dist.euclidean(upper_lip[upper_center_idx], lower_lip[lower_center_idx])

        # Calculate horizontal width
        horizontal_width = dist.euclidean(
            landmarks[self.LEFT_MOUTH_CORNER],
            landmarks[self.RIGHT_MOUTH_CORNER]
        )

        # Calculate mouth aspect ratio (vertical/horizontal)
        mouth_aspect = vertical_opening / horizontal_width if horizontal_width > 0 else 0

        # Calculate approximate mouth area using the shoelace formula
        # Combine upper and lower lip points to form a closed contour
        mouth_points = np.vstack([upper_lip[:, :2], lower_lip[::-1, :2]])
        mouth_area = 0.5 * abs(sum(
            mouth_points[i, 0] * mouth_points[i + 1, 1] - mouth_points[i + 1, 0] * mouth_points[i, 1]
            for i in range(len(mouth_points) - 1)
        ))

        # Calculate standard MAR
        mouth = landmarks[self.MOUTH_INDICES]
        mar = self.calculate_mar(mouth)

        return mar, vertical_opening, mouth_area, mouth_aspect

    def detect_yawn(self, mar, vertical_opening, mouth_area, mouth_aspect):
        """
        Improved yawn detection that differentiates from talking

        Yawn characteristics:
        - Large vertical mouth opening
        - Sustained duration (1-2 seconds)
        - Gradual increase and decrease in mouth opening
        - Higher vertical-to-horizontal ratio
        - Larger mouth area

        Talking characteristics:
        - Rapid mouth movements
        - Smaller vertical opening
        - Frequent changes in mouth shape
        - Lower vertical-to-horizontal ratio
        """
        # Yawn thresholds
        yawn_mar_threshold = 0.6  # High MAR for yawns
        yawn_vertical_threshold = 40  # Pixels (will vary by face size)
        yawn_aspect_threshold = 0.5  # Vertical opening is at least 50% of horizontal width
        yawn_min_duration = 20  # frames (~0.67 seconds at 30fps)
        yawn_max_duration = 90  # frames (~3 seconds at 30fps)

        # Check if mouth is significantly open
        mouth_significantly_open = (
            mar > yawn_mar_threshold and
            mouth_aspect > yawn_aspect_threshold and
            vertical_opening > yawn_vertical_threshold
        )

        if mouth_significantly_open:
            if self.consecutive_mouth_open_frames == 0:
                # Starting a potential yawn, record initial MAR
                self.mouth_open_start_mar = mar

            self.consecutive_mouth_open_frames += 1

            # Check if this is a sustained yawn (not rapid talking)
            if yawn_min_duration <= self.consecutive_mouth_open_frames <= yawn_max_duration:
                # This is likely a yawn - sustained opening
                if not self.last_yawn_detected:
                    self.last_yawn_detected = True
                    return True  # New yawn detected
        else:
            # Mouth closed or only slightly open
            if self.consecutive_mouth_open_frames > 0:
                # Mouth just closed
                # If it was open for too short (talking) or too long, it's not a yawn
                if self.consecutive_mouth_open_frames < yawn_min_duration:
                    # Too short - likely talking or other movement
                    pass

            # Reset counters
            self.consecutive_mouth_open_frames = 0
            self.mouth_open_start_mar = 0
            self.last_yawn_detected = False

        return False

    def calculate_pupil_size(self, iris_landmarks):
        """Calculate relative pupil size"""
        if len(iris_landmarks) < 4:
            return 0.0

        # Calculate iris diameter (approximation)
        horizontal = dist.euclidean(iris_landmarks[0], iris_landmarks[2])
        vertical = dist.euclidean(iris_landmarks[1], iris_landmarks[3])

        return (horizontal + vertical) / 2.0

    def calculate_gaze_direction(self, left_iris, right_iris, left_eye_corners, right_eye_corners):
        """Calculate gaze direction vector"""
        # Calculate iris center
        left_iris_center = np.mean(left_iris, axis=0)
        right_iris_center = np.mean(right_iris, axis=0)

        # Calculate eye center from corners
        left_eye_center = (left_eye_corners[0] + left_eye_corners[3]) / 2.0
        right_eye_center = (right_eye_corners[0] + right_eye_corners[3]) / 2.0

        # Calculate gaze offset
        left_gaze_offset = left_iris_center - left_eye_center
        right_gaze_offset = right_iris_center - right_eye_center

        avg_gaze = (left_gaze_offset + right_gaze_offset) / 2.0

        return avg_gaze

    def calculate_head_pose(self, landmarks, image_shape):
        """Calculate head pose angles (pitch, yaw, roll)"""
        h, w = image_shape[:2]

        # 3D model points (at least 6 points required for solvePnP)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # 2D image points
        image_points = np.array([
            [landmarks[self.NOSE_TIP][0], landmarks[self.NOSE_TIP][1]],
            [landmarks[self.CHIN][0], landmarks[self.CHIN][1]],
            [landmarks[self.LEFT_EYE_LEFT][0], landmarks[self.LEFT_EYE_LEFT][1]],
            [landmarks[self.RIGHT_EYE_RIGHT][0], landmarks[self.RIGHT_EYE_RIGHT][1]],
            [landmarks[self.LEFT_MOUTH_CORNER][0], landmarks[self.LEFT_MOUTH_CORNER][1]],
            [landmarks[self.RIGHT_MOUTH_CORNER][0], landmarks[self.RIGHT_MOUTH_CORNER][1]]
        ], dtype="double")

        # Camera internals
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0, 0, 0

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Calculate Euler angles
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = euler_angles.flatten()[:3]

        return pitch, yaw, roll

    def detect_microsleep(self):
        """Detect microsleep episodes (very brief sleep periods)"""
        if self.consecutive_closed_frames >= self.microsleep_frames:
            return True
        return False

    def calculate_perclos(self):
        """Calculate PERCLOS (Percentage of Eye Closure)"""
        if len(self.eye_closure_history) == 0:
            return 0.0

        closed_count = sum(self.eye_closure_history)
        total_count = len(self.eye_closure_history)

        return closed_count / total_count

    def calculate_blink_rate(self):
        """Calculate blinks per minute"""
        current_time = time.time()
        recent_blinks = [t for t in self.blink_timestamps if current_time - t < 60]
        return len(recent_blinks)

    def calculate_yawn_rate(self):
        """Calculate yawns per minute"""
        current_time = time.time()
        recent_yawns = [t for t in self.yawn_timestamps if current_time - t < 60]
        return len(recent_yawns)

    def calculate_fatigue_score(self, ear, mar, perclos, blink_rate, yawn_rate, head_pose):
        """
        Advanced multi-modal fatigue score calculation (0-100)
        Higher score = more fatigued
        """
        score = 0

        # PERCLOS contribution (0-35 points) - very conservative, ignore normal blinking
        if perclos > 0.60:  # Eyes closed more than 60% of time
            score += 35
        elif perclos > 0.50:
            score += 25
        elif perclos > 0.40:
            score += 15
        elif perclos > 0.30:
            score += 8
        # Below 30% is considered normal (includes regular blinking)

        # EAR contribution (0-25 points) - very conservative
        if self.baseline_ear:
            ear_deviation = (self.baseline_ear - ear) / self.baseline_ear
            if ear_deviation > 0.5:  # Eyes must be significantly more closed
                score += 25
            elif ear_deviation > 0.4:
                score += 15
            elif ear_deviation > 0.3:
                score += 8
            # Less than 30% deviation is normal (includes blinking)

        # Blink rate contribution (0-15 points) - more lenient
        if blink_rate < 3:  # Very few blinks (possible microsleep)
            score += 15
        elif blink_rate > 50:  # Excessive blinking (very high threshold)
            score += 10
        # Normal blink rate (3-50/min) contributes nothing

        # Yawn rate contribution (0-15 points)
        if yawn_rate > 6:
            score += 15
        elif yawn_rate > 3:
            score += 10
        elif yawn_rate > 1:
            score += 5

        # Head pose contribution (0-10 points)
        pitch, yaw, roll = head_pose
        if abs(pitch) > 20 or abs(yaw) > 25 or abs(roll) > 15:
            score += 10
        elif abs(pitch) > 10 or abs(yaw) > 15 or abs(roll) > 10:
            score += 5

        return min(score, 100)

    def determine_alert_level(self, fatigue_score, ear, perclos, microsleep):
        """
        Determine alert level with temporal smoothing
        Level 0: Normal (fatigue < 30)
        Level 1: Early warning (fatigue 30-50)
        Level 2: Moderate fatigue (fatigue 50-70)
        Level 3: High fatigue (fatigue 70-85)
        Level 4: Critical drowsiness (fatigue > 85 or microsleep detected)
        """
        # Instant alert level
        if microsleep or fatigue_score > 85:
            instant_level = 4
        elif fatigue_score > 70:
            instant_level = 3
        elif fatigue_score > 50:
            instant_level = 2
        elif fatigue_score > 30:
            instant_level = 1
        else:
            instant_level = 0

        # Add to history
        self.alert_history.append(instant_level)

        # Temporal smoothing with majority vote
        if len(self.alert_history) >= 20:  # ~0.67 seconds at 30fps
            recent_alerts = list(self.alert_history)[-20:]

            # Count occurrences of each level
            level_counts = {i: recent_alerts.count(i) for i in range(5)}

            # Determine smoothed level (prefer higher alerts for safety)
            if level_counts[4] >= 10:  # 50% critical
                return 4
            elif level_counts[3] >= 12:  # 60% high
                return 3
            elif level_counts[2] >= 14:  # 70% moderate
                return 2
            elif level_counts[1] >= 14:  # 70% warning
                return 1
            else:
                return 0

        return instant_level

    def calibrate(self, frame):
        """Add frame to calibration data"""
        landmarks = self._extract_landmarks(frame)
        if landmarks is None:
            return False

        # Extract calibration metrics
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        mouth = landmarks[self.MOUTH_INDICES]

        ear_left = self.calculate_ear(left_eye)
        ear_right = self.calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0

        mar = self.calculate_mar(mouth)

        self.calibration_data['ear_values'].append(ear)
        self.calibration_data['mar_values'].append(mar)

        return True

    def finalize_calibration(self):
        """Calculate baseline values from calibration data"""
        if len(self.calibration_data['ear_values']) < 30:
            return False

        # Use median for robustness
        self.baseline_ear = np.median(self.calibration_data['ear_values'])
        self.baseline_mar = np.median(self.calibration_data['mar_values'])

        # Adjust thresholds based on baseline (made more conservative)
        # Use 70% of baseline instead of 85% to avoid flagging normal blinks
        self.ear_threshold = self.baseline_ear * 0.70

        self.is_calibrated = True
        return True

    def _extract_landmarks(self, frame):
        """Extract facial landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark

        # Convert to numpy array with pixel coordinates
        landmarks_array = np.array([
            [lm.x * w, lm.y * h, lm.z * w] for lm in landmarks
        ])

        return landmarks_array

    def process_frame(self, frame):
        """
        Process a single frame and return detection results
        """
        self.total_frames += 1

        # Extract landmarks
        landmarks = self._extract_landmarks(frame)

        if landmarks is None:
            return None, frame

        # Extract feature points
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        mouth = landmarks[self.MOUTH_INDICES]

        # Calculate metrics
        ear_left = self.calculate_ear(left_eye)
        ear_right = self.calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0

        # Calculate detailed mouth metrics for better yawn detection
        mar, vertical_opening, mouth_area, mouth_aspect = self.calculate_mouth_metrics(landmarks)

        # Update histories
        self.ear_history.append(ear)
        self.mar_history.append(mar)

        # Eye closure detection
        eyes_closed = ear < self.ear_threshold
        self.eye_closure_history.append(1 if eyes_closed else 0)

        if eyes_closed:
            self.consecutive_closed_frames += 1
        else:
            if self.consecutive_closed_frames > 0:
                # Blink detected (increased minimum to filter out noise, kept max at 12 for normal blinks)
                if 4 <= self.consecutive_closed_frames <= 12:
                    self.blink_count += 1
                    self.blink_timestamps.append(time.time())
            self.consecutive_closed_frames = 0

        # Improved yawn detection (differentiates from talking)
        yawning = self.detect_yawn(mar, vertical_opening, mouth_area, mouth_aspect)
        if yawning:
            self.yawn_count += 1
            self.yawn_timestamps.append(time.time())

        # Microsleep detection
        microsleep = self.detect_microsleep()
        if microsleep:
            self.microsleep_count += 1

        # Calculate advanced metrics
        perclos = self.calculate_perclos()
        blink_rate = self.calculate_blink_rate()
        yawn_rate = self.calculate_yawn_rate()

        # Head pose
        head_pose = self.calculate_head_pose(landmarks, frame.shape)
        self.head_pose_history.append(head_pose)

        # Fatigue score
        fatigue_score = self.calculate_fatigue_score(
            ear, mar, perclos, blink_rate, yawn_rate, head_pose
        )

        # Alert level
        alert_level = self.determine_alert_level(fatigue_score, ear, perclos, microsleep)

        # Prepare metrics
        metrics = {
            'fatigue_score': round(fatigue_score, 2),
            'attention_score': round(100 - fatigue_score, 2),
            'ear': round(ear, 3),
            'mar': round(mar, 3),
            'mouth_aspect': round(mouth_aspect, 3),
            'mouth_open_frames': int(self.consecutive_mouth_open_frames),
            'perclos': round(perclos * 100, 2),
            'blink_rate': int(blink_rate),
            'yawn_rate': int(yawn_rate),
            'blink_count': int(self.blink_count),
            'yawn_count': int(self.yawn_count),
            'microsleep_count': int(self.microsleep_count),
            'alert_level': int(alert_level),
            'eyes_closed': bool(eyes_closed),
            'yawning': bool(yawning),
            'microsleep': bool(microsleep),
            'head_pose': {
                'pitch': round(float(head_pose[0]), 2),
                'yaw': round(float(head_pose[1]), 2),
                'roll': round(float(head_pose[2]), 2)
            },
            'elapsed_time': round(time.time() - self.start_time, 1),
            'fps': round(self.total_frames / (time.time() - self.start_time), 1),
            'is_calibrated': bool(self.is_calibrated)
        }

        # Draw visualization
        annotated_frame = self._draw_annotations(frame, landmarks, metrics)

        return metrics, annotated_frame

    def _draw_annotations(self, frame, landmarks, metrics):
        """Draw detection results on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw face mesh
        for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
            x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
            cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

        # Draw upper lip contour (change color if yawning)
        mouth_color = (0, 0, 255) if metrics['yawning'] else (0, 255, 255)  # Red if yawning, cyan otherwise
        upper_lip_points = np.array([[int(landmarks[idx][0]), int(landmarks[idx][1])]
                                     for idx in self.UPPER_LIP_INDICES], dtype=np.int32)
        cv2.polylines(annotated, [upper_lip_points], False, mouth_color, 2)

        # Draw lower lip contour (change color if yawning)
        lower_lip_points = np.array([[int(landmarks[idx][0]), int(landmarks[idx][1])]
                                     for idx in self.LOWER_LIP_INDICES], dtype=np.int32)
        cv2.polylines(annotated, [lower_lip_points], False, mouth_color, 2)

        # Draw mouth corner points for clarity
        for idx in [self.LEFT_MOUTH_CORNER, self.RIGHT_MOUTH_CORNER]:
            x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
            cv2.circle(annotated, (x, y), 3, (255, 255, 0), -1)

        # Draw yawn indicator if yawning
        if metrics['yawning']:
            yawn_text = "YAWNING!"
            text_size = cv2.getTextSize(yawn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            # Position near mouth
            mouth_center_x = int((landmarks[self.LEFT_MOUTH_CORNER][0] + landmarks[self.RIGHT_MOUTH_CORNER][0]) / 2)
            mouth_bottom_y = int(max(landmarks[idx][1] for idx in self.LOWER_LIP_INDICES))
            text_pos = (mouth_center_x - text_size[0] // 2, mouth_bottom_y + 30)
            cv2.putText(annotated, yawn_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Alert color
        alert_colors = [
            (0, 255, 0),   # Green - Normal
            (0, 255, 255), # Yellow - Warning
            (0, 165, 255), # Orange - Moderate
            (0, 100, 255), # Dark Orange - High
            (0, 0, 255)    # Red - Critical
        ]
        alert_color = alert_colors[metrics['alert_level']]

        # Draw border
        cv2.rectangle(annotated, (0, 0), (w-1, h-1), alert_color, 8)

        # Info panel
        panel_height = 180
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # Text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 25
        line_height = 22

        texts = [
            f"Fatigue Score: {metrics['fatigue_score']:.1f}%",
            f"Alert Level: {metrics['alert_level']} {'CRITICAL' if metrics['alert_level'] >= 4 else 'HIGH' if metrics['alert_level'] >= 3 else 'MODERATE' if metrics['alert_level'] >= 2 else 'WARNING' if metrics['alert_level'] >= 1 else 'NORMAL'}",
            f"EAR: {metrics['ear']:.3f} | MAR: {metrics['mar']:.3f}",
            f"PERCLOS: {metrics['perclos']:.1f}%",
            f"Blinks: {metrics['blink_count']} ({metrics['blink_rate']}/min)",
            f"Yawns: {metrics['yawn_count']} ({metrics['yawn_rate']}/min)",
            f"Microsleeps: {metrics['microsleep_count']}",
            f"FPS: {metrics['fps']:.1f}"
        ]

        for i, text in enumerate(texts):
            cv2.putText(annotated, text, (10, y_offset + i * line_height),
                       font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Status indicator
        status_text = "ALERT!" if metrics['alert_level'] >= 3 else "Warning" if metrics['alert_level'] >= 1 else "Normal"
        text_size = cv2.getTextSize(status_text, font, 1.0, 2)[0]
        text_x = w - text_size[0] - 20
        cv2.putText(annotated, status_text, (text_x, 40),
                   font, 1.0, alert_color, 2, cv2.LINE_AA)

        return annotated

    def release(self):
        """Cleanup resources"""
        self.face_mesh.close()
