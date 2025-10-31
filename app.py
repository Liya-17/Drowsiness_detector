"""
Advanced Drowsiness Detection System - Web Application
Real-time monitoring with modern dashboard and analytics
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import yaml
import threading
import time
import json
import os
from datetime import datetime
from collections import deque

from advanced_detector import AdvancedDrowsinessDetector
from alert_system import AlertSystem, VisualAlertOverlay


app = Flask(__name__)
app.config['SECRET_KEY'] = 'drowsiness-detection-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Global variables
detector = None
alert_system = None
visual_overlay = None
camera = None
is_running = False
is_calibrating = False
calibration_progress = 0

# Metrics history for charts
metrics_history = deque(maxlen=900)  # 30 seconds at 30fps

# Session management
current_session = None


def initialize_system():
    """Initialize detection system and camera"""
    global detector, alert_system, visual_overlay, camera

    detector = AdvancedDrowsinessDetector(config)
    alert_system = AlertSystem(config)
    visual_overlay = VisualAlertOverlay()

    # Open camera
    camera_id = config['camera']['id']
    print(f"Opening camera {camera_id}...")
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    camera.set(cv2.CAP_PROP_FPS, config['camera']['fps'])

    if not camera.isOpened():
        raise Exception(f"Could not open camera {camera_id}")

    print(f"Camera {camera_id} opened successfully")


def generate_frames():
    """Generate video frames with detection results"""
    global is_running, metrics_history

    while is_running:
        try:
            success, frame = camera.read()
            if not success:
                print("Failed to read from camera")
                break

            # Process frame
            metrics, annotated_frame = detector.process_frame(frame)

            if metrics:
                # Add to history
                metrics_history.append(metrics)

                # Check for alerts
                try:
                    alert = alert_system.check_and_alert(metrics['alert_level'], metrics)
                    if alert:
                        visual_overlay.set_alert(alert)
                        # Emit alert to clients
                        socketio.emit('alert', alert)
                except Exception as e:
                    print(f"Alert system error: {e}")

                # Draw visual overlay
                try:
                    annotated_frame = visual_overlay.draw(annotated_frame)
                except Exception as e:
                    print(f"Visual overlay error: {e}")

                # Emit metrics to clients
                try:
                    socketio.emit('metrics_update', metrics)
                except Exception as e:
                    print(f"Metrics emit error: {e}")

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print("Failed to encode frame")
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Small delay to control frame rate
            time.sleep(1 / config['camera']['fps'])

        except Exception as e:
            print(f"Frame generation error: {e}")
            import traceback
            traceback.print_exc()
            break


def calibration_routine():
    """Perform user calibration"""
    global is_calibrating, calibration_progress, detector

    is_calibrating = True
    calibration_progress = 0

    calibration_duration = config['calibration']['duration']
    frames_needed = calibration_duration * config['camera']['fps']
    frames_collected = 0

    socketio.emit('calibration_status', {
        'status': 'started',
        'progress': 0,
        'message': 'Please sit naturally and look at the camera'
    })

    start_time = time.time()

    while frames_collected < frames_needed and is_calibrating:
        success, frame = camera.read()
        if not success:
            continue

        # Add to calibration
        if detector.calibrate(frame):
            frames_collected += 1
            calibration_progress = int((frames_collected / frames_needed) * 100)

            # Send progress update
            socketio.emit('calibration_status', {
                'status': 'in_progress',
                'progress': calibration_progress,
                'message': f'Calibrating... {calibration_progress}%'
            })

        time.sleep(1 / config['camera']['fps'])

    # Finalize calibration
    if detector.finalize_calibration():
        socketio.emit('calibration_status', {
            'status': 'completed',
            'progress': 100,
            'message': 'Calibration complete! System is now personalized.',
            'baseline_ear': detector.baseline_ear,
            'baseline_mar': detector.baseline_mar
        })
    else:
        socketio.emit('calibration_status', {
            'status': 'failed',
            'progress': 0,
            'message': 'Calibration failed. Please try again.'
        })

    is_calibrating = False


@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start drowsiness detection"""
    global is_running, current_session

    if is_running:
        return jsonify({'status': 'error', 'message': 'Already running'})

    # Check if calibration is required
    if config['calibration']['require_on_startup'] and not detector.is_calibrated:
        return jsonify({
            'status': 'error',
            'message': 'Calibration required. Please calibrate first.'
        })

    is_running = True

    # Start new session
    current_session = {
        'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'start_time': datetime.now().isoformat(),
        'events': []
    }

    # Start frame generation in background
    threading.Thread(target=generate_frames, daemon=True).start()

    return jsonify({'status': 'success', 'message': 'Detection started'})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop drowsiness detection"""
    global is_running, current_session

    if not is_running:
        return jsonify({'status': 'error', 'message': 'Not running'})

    is_running = False

    # Save session data
    if current_session and config['logging']['save_sessions']:
        save_session_data()

    return jsonify({'status': 'success', 'message': 'Detection stopped'})


@app.route('/api/calibrate', methods=['POST'])
def start_calibration():
    """Start calibration process"""
    global is_calibrating

    if is_calibrating:
        return jsonify({'status': 'error', 'message': 'Calibration already in progress'})

    if is_running:
        return jsonify({'status': 'error', 'message': 'Stop detection first'})

    # Start calibration in background thread
    threading.Thread(target=calibration_routine, daemon=True).start()

    return jsonify({'status': 'success', 'message': 'Calibration started'})


@app.route('/api/cancel_calibration', methods=['POST'])
def cancel_calibration():
    """Cancel ongoing calibration"""
    global is_calibrating

    is_calibrating = False
    return jsonify({'status': 'success', 'message': 'Calibration cancelled'})


@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    if len(metrics_history) > 0:
        return jsonify(metrics_history[-1])
    return jsonify({})


@app.route('/api/metrics/history')
def get_metrics_history():
    """Get metrics history for charts"""
    # Return last N data points
    limit = int(request.args.get('limit', 100))
    history = list(metrics_history)[-limit:]

    return jsonify({
        'timestamps': [m['elapsed_time'] for m in history],
        'fatigue_scores': [m['fatigue_score'] for m in history],
        'ear_values': [m['ear'] for m in history],
        'perclos_values': [m['perclos'] for m in history],
        'alert_levels': [m['alert_level'] for m in history]
    })


@app.route('/api/statistics')
def get_statistics():
    """Get session statistics"""
    if len(metrics_history) == 0:
        return jsonify({})

    recent_metrics = list(metrics_history)

    stats = {
        'session_duration': recent_metrics[-1]['elapsed_time'] if recent_metrics else 0,
        'total_blinks': recent_metrics[-1]['blink_count'] if recent_metrics else 0,
        'total_yawns': recent_metrics[-1]['yawn_count'] if recent_metrics else 0,
        'total_microsleeps': recent_metrics[-1]['microsleep_count'] if recent_metrics else 0,
        'avg_fatigue': sum(m['fatigue_score'] for m in recent_metrics) / len(recent_metrics),
        'max_fatigue': max(m['fatigue_score'] for m in recent_metrics),
        'avg_perclos': sum(m['perclos'] for m in recent_metrics) / len(recent_metrics),
        'alert_distribution': {
            '0': sum(1 for m in recent_metrics if m['alert_level'] == 0),
            '1': sum(1 for m in recent_metrics if m['alert_level'] == 1),
            '2': sum(1 for m in recent_metrics if m['alert_level'] == 2),
            '3': sum(1 for m in recent_metrics if m['alert_level'] == 3),
            '4': sum(1 for m in recent_metrics if m['alert_level'] == 4)
        }
    }

    return jsonify(stats)


@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration"""
    global config

    if request.method == 'POST':
        # Update configuration
        updates = request.json
        # Merge updates into config
        config.update(updates)

        # Save to file
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)

        return jsonify({'status': 'success', 'message': 'Configuration updated'})

    return jsonify(config)


def save_session_data():
    """Save session data to file"""
    if not current_session:
        return

    # Create sessions directory
    session_dir = config['logging']['save_directory']
    os.makedirs(session_dir, exist_ok=True)

    session_id = current_session['id']

    # Prepare session data
    session_data = {
        'session_id': session_id,
        'start_time': current_session['start_time'],
        'end_time': datetime.now().isoformat(),
        'metrics_history': list(metrics_history),
        'alert_statistics': alert_system.get_alert_statistics(),
        'configuration': config
    }

    # Save as JSON
    if 'json' in config['logging']['export_format']:
        json_path = os.path.join(session_dir, f'session_{session_id}.json')
        with open(json_path, 'w') as f:
            json.dump(session_data, f, indent=2)

    # Save as CSV
    if 'csv' in config['logging']['export_format']:
        import pandas as pd
        csv_path = os.path.join(session_dir, f'session_{session_id}.csv')
        df = pd.DataFrame(list(metrics_history))
        df.to_csv(csv_path, index=False)


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass


def cleanup():
    """Cleanup resources on shutdown"""
    global camera, detector, alert_system

    if camera:
        camera.release()
    if detector:
        detector.release()
    if alert_system:
        alert_system.cleanup()


if __name__ == '__main__':
    try:
        print("Initializing Advanced Drowsiness Detection System...")
        initialize_system()
        print(f"System initialized successfully!")
        print(f"Starting web server on http://{config['dashboard']['host']}:{config['dashboard']['port']}")
        print("\nPress Ctrl+C to stop")

        socketio.run(app,
                    host=config['dashboard']['host'],
                    port=config['dashboard']['port'],
                    debug=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup()
    except Exception as e:
        print(f"Error: {e}")
        cleanup()
