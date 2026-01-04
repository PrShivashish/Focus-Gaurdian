import cv2

print("Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ CAMERA FAILED")
    exit(1)

print("✅ Camera works!")

for i in range(5):
    ret, frame = cap.read()
    print(f"Frame {i+1}: {'✅ OK' if ret else '❌ FAIL'}")

cap.release()
