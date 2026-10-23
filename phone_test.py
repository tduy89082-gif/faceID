import cv2
import threading
from insightface.app import FaceAnalysis


PHONE_CAMERA_URL = "https://53.254.45.242:8080/video"


    

class CameraStream:

    def __init__(self, url):

        self.camera = cv2.VideoCapture(url)

        if not self.camera.isOpened():
            raise RuntimeError("Cannot connect to phone camera")

        # Cố gắng giảm buffer
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.camera.read()

        if not self.ret:
            raise RuntimeError("Cannot read first frame")

        self.running = True

        # Lock để tránh đọc/ghi frame cùng lúc
        self.lock = threading.Lock()

        # Thread riêng để liên tục lấy frame mới
        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()




    def update(self):

        while self.running:

            ret, frame = self.camera.read()

            if ret:

                with self.lock:

                    self.ret = ret
                    self.frame = frame.copy()


  

    def read(self):

        with self.lock:

            if self.frame is None:
                return False, None

            return self.ret, self.frame.copy()


    

    def stop(self):

        self.running = False

        self.thread.join(timeout=1)

        self.camera.release()




print("Loading FaceID model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("Model loaded")




camera = CameraStream(PHONE_CAMERA_URL)




while True:

    ret, frame = camera.read()

    if not ret:
        print("Cannot read frame")
        break


   

    faces = app.get(frame)


    for face in faces:

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )



    cv2.imshow(
        "FaceID",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) == ord("q"):
        break




camera.stop()

cv2.destroyAllWindows()