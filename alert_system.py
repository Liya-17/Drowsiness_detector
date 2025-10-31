"""
Advanced Alert System with Multi-Level Graduated Warnings
Includes audio alerts, visual alerts, and haptic feedback support
"""

import cv2
import pygame
import numpy as np
import time
from collections import deque


class AlertSystem:
    """
    Multi-level alert system with adaptive alerting
    Prevents alert fatigue through intelligent timing
    """

    ALERT_LEVELS = {
        0: {"name": "Normal", "color": (0, 255, 0), "priority": 0},
        1: {"name": "Early Warning", "color": (0, 255, 255), "priority": 1},
        2: {"name": "Moderate Fatigue", "color": (0, 165, 255), "priority": 2},
        3: {"name": "High Fatigue", "color": (0, 100, 255), "priority": 3},
        4: {"name": "Critical Drowsiness", "color": (0, 0, 255), "priority": 4}
    }

    def __init__(self, config):
        """Initialize alert system"""
        self.config = config
        self.audio_enabled = config['audio']['enabled']

        # Initialize pygame for audio
        if self.audio_enabled:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

        # Alert cooldowns (seconds)
        self.cooldowns = {
            1: config['alerts']['cooldown_level_1'],
            2: config['alerts']['cooldown_level_2'],
            3: config['alerts']['cooldown_level_3'],
            4: config['alerts']['cooldown_level_4']
        }

        # Last alert times
        self.last_alert_times = {1: 0, 2: 0, 3: 0, 4: 0}

        # Alert history for adaptive behavior
        self.alert_history = deque(maxlen=1000)
        self.dismissed_count = 0

        # Volume
        self.volume = config['audio']['volume']

        # Pre-generate alert sounds
        self.alert_sounds = {}
        if self.audio_enabled:
            self._generate_alert_sounds()

    def _generate_alert_sounds(self):
        """Generate different alert sounds for each level"""
        sample_rate = 22050

        # Level 1: Gentle beep
        self.alert_sounds[1] = self._generate_beep(600, 0.15, sample_rate, smooth=True)

        # Level 2: Double beep
        beep1 = self._generate_beep(700, 0.12, sample_rate, smooth=True)
        silence = np.zeros((int(0.08 * sample_rate), 2), dtype=np.int16)
        beep2 = self._generate_beep(700, 0.12, sample_rate, smooth=True)
        self.alert_sounds[2] = np.concatenate([beep1, silence, beep2])

        # Level 3: Rising tone
        self.alert_sounds[3] = self._generate_sweep(700, 1000, 0.4, sample_rate)

        # Level 4: Urgent alarm
        pulse1 = self._generate_beep(1200, 0.15, sample_rate, smooth=False)
        silence = np.zeros((int(0.05 * sample_rate), 2), dtype=np.int16)
        pulse2 = self._generate_beep(1200, 0.15, sample_rate, smooth=False)
        silence2 = np.zeros((int(0.05 * sample_rate), 2), dtype=np.int16)
        pulse3 = self._generate_beep(1200, 0.15, sample_rate, smooth=False)
        self.alert_sounds[4] = np.concatenate([pulse1, silence, pulse2, silence2, pulse3])

    def _generate_beep(self, frequency, duration, sample_rate, smooth=True):
        """Generate a beep tone"""
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, False)

        # Generate sine wave
        wave = np.sin(2 * np.pi * frequency * t)

        # Apply envelope for smooth attack and decay
        if smooth:
            envelope_samples = int(0.02 * sample_rate)  # 20ms attack/decay
            attack = np.linspace(0, 1, envelope_samples)
            decay = np.linspace(1, 0, envelope_samples)
            sustain = np.ones(samples - 2 * envelope_samples)
            envelope = np.concatenate([attack, sustain, decay])
            wave = wave * envelope

        # Convert to 16-bit PCM
        wave = (wave * 32767 * self.volume).astype(np.int16)

        # Stereo
        wave = np.repeat(wave.reshape(-1, 1), 2, axis=1)

        return wave

    def _generate_sweep(self, start_freq, end_freq, duration, sample_rate):
        """Generate a frequency sweep"""
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, False)

        # Frequency sweep
        freq = np.linspace(start_freq, end_freq, samples)
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate

        wave = np.sin(phase)

        # Apply envelope
        envelope_samples = int(0.02 * sample_rate)
        attack = np.linspace(0, 1, envelope_samples)
        decay = np.linspace(1, 0, envelope_samples)
        sustain = np.ones(samples - 2 * envelope_samples)
        envelope = np.concatenate([attack, sustain, decay])
        wave = wave * envelope

        # Convert to 16-bit PCM
        wave = (wave * 32767 * self.volume).astype(np.int16)

        # Stereo
        wave = np.repeat(wave.reshape(-1, 1), 2, axis=1)

        return wave

    def _play_sound(self, sound_array):
        """Play a sound from numpy array"""
        sound = pygame.sndarray.make_sound(sound_array)
        sound.play()

    def check_and_alert(self, alert_level, metrics):
        """
        Check if alert should be triggered and play appropriate sound
        Returns alert message or None
        """
        current_time = time.time()

        # No alert for normal level
        if alert_level == 0:
            return None

        # Check cooldown
        last_time = self.last_alert_times.get(alert_level, 0)
        cooldown = self.cooldowns.get(alert_level, 10)

        if current_time - last_time < cooldown:
            return None

        # Update last alert time
        self.last_alert_times[alert_level] = current_time

        # Record alert
        self.alert_history.append({
            'level': alert_level,
            'time': current_time,
            'metrics': metrics
        })

        # Create alert message
        alert_info = self.ALERT_LEVELS[alert_level]
        message = self._create_alert_message(alert_level, metrics)

        # Play audio alert
        if self.audio_enabled and alert_level in self.alert_sounds:
            self._play_sound(self.alert_sounds[alert_level])

        return {
            'level': alert_level,
            'name': alert_info['name'],
            'message': message,
            'color': alert_info['color'],
            'priority': alert_info['priority'],
            'timestamp': current_time
        }

    def _create_alert_message(self, level, metrics):
        """Create detailed alert message based on metrics"""
        messages = {
            1: "Early signs of fatigue detected. Consider taking a break soon.",
            2: "Moderate fatigue detected. You should take a break.",
            3: "High fatigue! Take a break immediately.",
            4: "CRITICAL: Severe drowsiness detected! Stop and rest NOW!"
        }

        base_message = messages.get(level, "Alert!")

        # Add specific reasons
        reasons = []
        if metrics.get('perclos', 0) > 30:
            reasons.append(f"Eye closure: {metrics['perclos']:.1f}%")
        if metrics.get('yawn_rate', 0) > 3:
            reasons.append(f"Frequent yawning: {metrics['yawn_rate']}/min")
        if metrics.get('microsleep', False):
            reasons.append("Microsleep detected")
        if metrics.get('blink_rate', 15) < 5:
            reasons.append("Reduced blinking (staring)")

        if reasons:
            base_message += " Indicators: " + ", ".join(reasons)

        return base_message

    def get_alert_statistics(self):
        """Get statistics about alerts"""
        if len(self.alert_history) == 0:
            return None

        recent_alerts = list(self.alert_history)[-100:]  # Last 100 alerts

        level_counts = {i: 0 for i in range(5)}
        for alert in recent_alerts:
            level_counts[alert['level']] += 1

        avg_time_between_alerts = 0
        if len(recent_alerts) > 1:
            times = [a['time'] for a in recent_alerts]
            time_diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_time_between_alerts = np.mean(time_diffs)

        return {
            'total_alerts': len(self.alert_history),
            'recent_alerts': len(recent_alerts),
            'level_distribution': level_counts,
            'avg_time_between_alerts': avg_time_between_alerts,
            'last_alert_time': recent_alerts[-1]['time'] if recent_alerts else None
        }

    def adaptive_adjust_cooldowns(self):
        """
        Adaptively adjust cooldowns based on alert frequency
        Prevents alert fatigue
        """
        stats = self.get_alert_statistics()
        if not stats:
            return

        # If too many alerts in short time, increase cooldowns
        if stats['avg_time_between_alerts'] < 30 and stats['recent_alerts'] > 20:
            for level in self.cooldowns:
                self.cooldowns[level] = min(self.cooldowns[level] * 1.2, 60)
        # If alerts are rare, decrease cooldowns for responsiveness
        elif stats['avg_time_between_alerts'] > 120:
            for level in self.cooldowns:
                self.cooldowns[level] = max(self.cooldowns[level] * 0.8, 1)

    def reset(self):
        """Reset alert system"""
        self.last_alert_times = {1: 0, 2: 0, 3: 0, 4: 0}
        self.alert_history.clear()

    def cleanup(self):
        """Cleanup resources"""
        if self.audio_enabled:
            pygame.mixer.quit()


class VisualAlertOverlay:
    """
    Visual alert overlay for display
    Shows prominent warnings on screen
    """

    def __init__(self):
        """Initialize visual overlay"""
        self.current_alert = None
        self.alert_start_time = 0
        self.display_duration = 3.0  # seconds

    def set_alert(self, alert_info):
        """Set current alert to display"""
        if alert_info:
            self.current_alert = alert_info
            self.alert_start_time = time.time()

    def draw(self, frame):
        """Draw alert overlay on frame"""
        if not self.current_alert:
            return frame

        # Check if alert should still be displayed
        if time.time() - self.alert_start_time > self.display_duration:
            self.current_alert = None
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Pulsing effect
        elapsed = time.time() - self.alert_start_time
        pulse = 0.5 + 0.5 * np.sin(elapsed * 6)  # Fast pulse
        alpha = 0.3 + 0.3 * pulse

        # Draw semi-transparent banner
        banner_height = 100
        color = self.current_alert['color']

        cv2.rectangle(overlay, (0, h - banner_height), (w, h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw alert text
        font = cv2.FONT_HERSHEY_DUPLEX  # Using DUPLEX for a bold-like appearance
        text = self.current_alert['name'].upper()
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - banner_height + 45

        # Text with outline
        cv2.putText(frame, text, (text_x, text_y), font, 1.5, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Sub-text with message
        message = self.current_alert.get('message', '')
        if len(message) > 80:
            message = message[:77] + "..."

        text_size_msg = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        msg_x = (w - text_size_msg[0]) // 2
        msg_y = text_y + 30

        cv2.putText(frame, message, (msg_x, msg_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
