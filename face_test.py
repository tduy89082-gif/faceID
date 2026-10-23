import cv2
from insightface.app import FaceAnalysis


print("Loading model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Model loaded")


camera = cv2.VideoCapture(
    "http://192.168.1.11:8080/video"
)

if not camera.isOpened():
    print("Cannot open camera")
    exit()


while True:
    ret, frame = camera.read()

    if not ret:
        print("Cannot read frame")
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()