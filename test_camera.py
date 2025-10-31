"""
Simple camera test script
"""
import cv2

print("Testing camera access...")

# Try to open camera 0
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera 0")
    print("Trying camera 1...")
    camera = cv2.VideoCapture(1)

    if not camera.isOpened():
        print("ERROR: Could not open camera 1 either")
        exit(1)
    else:
        print("SUCCESS: Camera 1 opened!")
else:
    print("SUCCESS: Camera 0 opened!")

# Try to read a frame
success, frame = camera.read()

if not success:
    print("ERROR: Could not read frame from camera")
    camera.release()
    exit(1)

print(f"SUCCESS: Read frame with shape {frame.shape}")

# Save test image
cv2.imwrite("test_frame.jpg", frame)
print("Test frame saved as test_frame.jpg")

camera.release()
print("\nCamera test completed successfully!")
print("You can now run the main app: python app.py")
