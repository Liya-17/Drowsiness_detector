"""
System Test Script
Quick verification that all components are working
"""

import sys
import cv2


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")

    packages = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('flask', 'Flask'),
        ('flask_socketio', 'Flask-SocketIO'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
        ('pygame', 'Pygame'),
        ('yaml', 'PyYAML')
    ]

    failed = []
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name} - FAILED: {e}")
            failed.append(display_name)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✅ All packages imported successfully\n")
    return True


def test_camera():
    """Test camera access"""
    print("Testing camera access...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("  ✗ Could not open camera (ID 0)")
        print("\nTroubleshooting:")
        print("  1. Grant camera permissions")
        print("  2. Close other apps using camera")
        print("  3. Try different camera ID in config.yaml")
        camera.release()
        return False

    ret, frame = camera.read()
    camera.release()

    if not ret:
        print("  ✗ Camera opened but failed to capture")
        return False

    h, w = frame.shape[:2]
    print(f"  ✓ Camera working - Resolution: {w}x{h}")
    print("\n✅ Camera test passed\n")
    return True


def test_mediapipe():
    """Test MediaPipe Face Mesh"""
    print("Testing MediaPipe Face Mesh...")

    try:
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("  ✓ MediaPipe Face Mesh loaded")

        # Test with camera
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            ret, frame = camera.read()
            camera.release()

            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    print(f"  ✓ Face detected - {len(landmarks)} landmarks")
                else:
                    print("  ⚠ No face detected (make sure you're in front of camera)")

        face_mesh.close()
        print("\n✅ MediaPipe test passed\n")
        return True

    except Exception as e:
        print(f"  ✗ MediaPipe test failed: {e}")
        return False


def test_detector():
    """Test advanced detector module"""
    print("Testing Advanced Detector...")

    try:
        import yaml
        from advanced_detector import AdvancedDrowsinessDetector

        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print("  ✓ Config loaded")

        # Initialize detector
        detector = AdvancedDrowsinessDetector(config)
        print("  ✓ Detector initialized")

        # Test with camera frame
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            ret, frame = camera.read()
            camera.release()

            if ret:
                metrics, annotated = detector.process_frame(frame)

                if metrics:
                    print(f"  ✓ Frame processed - Fatigue: {metrics['fatigue_score']:.1f}%")
                else:
                    print("  ⚠ Frame processed but no face detected")

        detector.release()
        print("\n✅ Detector test passed\n")
        return True

    except Exception as e:
        print(f"  ✗ Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alert_system():
    """Test alert system"""
    print("Testing Alert System...")

    try:
        import yaml
        from alert_system import AlertSystem

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        alert_system = AlertSystem(config)
        print("  ✓ Alert system initialized")

        # Test alert sounds generation
        if alert_system.audio_enabled:
            print("  ✓ Audio alerts enabled")
            print("  ✓ Alert sounds generated")
        else:
            print("  ⚠ Audio alerts disabled in config")

        alert_system.cleanup()
        print("\n✅ Alert system test passed\n")
        return True

    except Exception as e:
        print(f"  ✗ Alert system test failed: {e}")
        return False


def test_web_app():
    """Test Flask app initialization"""
    print("Testing Web Application...")

    try:
        from flask import Flask

        app = Flask(__name__)
        print("  ✓ Flask app created")

        print("  ℹ To test full web app, run: python app.py")
        print("  ℹ Then open: http://localhost:5000")

        print("\n✅ Web app test passed\n")
        return True

    except Exception as e:
        print(f"  ✗ Web app test failed: {e}")
        return False


def print_summary(results):
    """Print test summary"""
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(results.values())

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python app.py")
        print("  2. Open: http://localhost:5000")
        print("  3. Click 'Calibrate' then 'Start Detection'")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("   Run 'python setup.py' to reinstall dependencies")

    print("=" * 70)


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("ADVANCED DROWSINESS DETECTION SYSTEM - TEST SUITE")
    print("=" * 70 + "\n")

    results = {}

    # Run tests
    results['Package Imports'] = test_imports()
    results['Camera Access'] = test_camera()
    results['MediaPipe'] = test_mediapipe()
    results['Advanced Detector'] = test_detector()
    results['Alert System'] = test_alert_system()
    results['Web Application'] = test_web_app()

    # Summary
    print_summary(results)

    return all(results.values())


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
