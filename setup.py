"""
Setup and Installation Script
Automated system setup with dependency checks and configuration
"""

import subprocess
import sys
import os
import platform


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def check_python_version():
    """Check if Python version is compatible"""
    print_info("Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False

    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip():
    """Check if pip is installed"""
    print_info("Checking pip installation...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      check=True, capture_output=True)
        print_success("pip is installed")
        return True
    except subprocess.CalledProcessError:
        print_error("pip is not installed")
        return False


def install_dependencies():
    """Install required Python packages"""
    print_header("Installing Dependencies")

    print_info("Installing packages from requirements.txt...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        print_success("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def verify_installations():
    """Verify critical packages are installed"""
    print_header("Verifying Installation")

    critical_packages = [
        'cv2',
        'mediapipe',
        'numpy',
        'flask',
        'sklearn',
        'pandas'
    ]

    all_ok = True
    for package in critical_packages:
        try:
            __import__(package)
            print_success(f"{package} - OK")
        except ImportError:
            print_error(f"{package} - MISSING")
            all_ok = False

    return all_ok


def check_camera():
    """Check if camera is accessible"""
    print_header("Camera Check")

    print_info("Testing camera access...")
    try:
        import cv2
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print_warning("Could not access default camera (ID 0)")
            print_info("You may need to:")
            print_info("  1. Grant camera permissions to Python")
            print_info("  2. Check if another application is using the camera")
            print_info("  3. Try a different camera ID in config.yaml")
            camera.release()
            return False

        ret, frame = camera.read()
        camera.release()

        if ret:
            print_success("Camera is accessible and working")
            return True
        else:
            print_warning("Camera opened but failed to capture frame")
            return False

    except Exception as e:
        print_error(f"Camera test failed: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")

    directories = [
        'sessions',
        'static',
        'static/css',
        'static/js',
        'templates'
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print_success(f"Created: {directory}/")
        else:
            print_info(f"Already exists: {directory}/")

    return True


def test_mediapipe():
    """Test MediaPipe Face Mesh"""
    print_header("Testing MediaPipe")

    print_info("Loading MediaPipe Face Mesh model...")
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh

        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print_success("MediaPipe Face Mesh loaded successfully")
        face_mesh.close()
        return True

    except Exception as e:
        print_error(f"MediaPipe test failed: {e}")
        return False


def check_config():
    """Check if config file exists"""
    print_header("Configuration Check")

    if os.path.exists('config.yaml'):
        print_success("config.yaml found")
        return True
    else:
        print_warning("config.yaml not found")
        print_info("Please ensure config.yaml is in the project directory")
        return False


def print_next_steps():
    """Print next steps for user"""
    print_header("Setup Complete!")

    print(f"{Colors.OKGREEN}✓ System is ready to use!{Colors.ENDC}\n")

    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"1. Review and customize settings in {Colors.OKCYAN}config.yaml{Colors.ENDC}")
    print(f"2. Start the system: {Colors.OKCYAN}python app.py{Colors.ENDC}")
    print(f"3. Open browser: {Colors.OKCYAN}http://localhost:5000{Colors.ENDC}")
    print(f"4. Click {Colors.OKCYAN}Calibrate{Colors.ENDC} button (first time only)")
    print(f"5. Click {Colors.OKCYAN}Start Detection{Colors.ENDC} to begin monitoring\n")

    print(f"{Colors.BOLD}Documentation:{Colors.ENDC}")
    print(f"  - Full guide: {Colors.OKCYAN}README.md{Colors.ENDC}")
    print(f"  - Configuration: {Colors.OKCYAN}config.yaml{Colors.ENDC}\n")

    print(f"{Colors.BOLD}Troubleshooting:{Colors.ENDC}")
    print(f"  - Camera issues: Check permissions and config.yaml camera ID")
    print(f"  - Import errors: Run {Colors.OKCYAN}pip install -r requirements.txt{Colors.ENDC}")
    print(f"  - Port in use: Change port in config.yaml dashboard section\n")


def main():
    """Main setup routine"""
    print_header("Advanced Drowsiness Detection System - Setup")

    print(f"{Colors.BOLD}System Information:{Colors.ENDC}")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")
    print()

    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("pip", check_pip),
    ]

    for name, check_func in checks:
        if not check_func():
            print_error(f"Setup failed at: {name}")
            print_info("Please fix the above issue and run setup again")
            return False

    # Install dependencies
    if not install_dependencies():
        return False

    # Verify installations
    if not verify_installations():
        print_error("Some packages failed to install")
        print_info("Try running: pip install -r requirements.txt")
        return False

    # Create directories
    create_directories()

    # Check config
    check_config()

    # Test MediaPipe
    test_mediapipe()

    # Check camera
    check_camera()

    # Print next steps
    print_next_steps()

    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
