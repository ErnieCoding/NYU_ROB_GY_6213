import cv2

url = "http://192.168.0.200:81/stream"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    cv2.imshow("ESP32-CAM", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()