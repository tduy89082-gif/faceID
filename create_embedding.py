import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis


PHONE_CAMERA_URL = "http://192.168.1.11:8080/video"

EMBEDDING_FILE = "registered_embedding.npy"

SAMPLE_COUNT = 15




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




camera = cv2.VideoCapture(PHONE_CAMERA_URL)

if not camera.isOpened():
    print("Cannot connect to phone camera")
    exit()


# Try to reduce camera buffer
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


embeddings = []

print()
print("====================================")
print(" FACE REGISTRATION")
print("====================================")
print("Look at the camera")
print("Keep your face inside the frame")
print("Collecting face samples...")
print()


while len(embeddings) < SAMPLE_COUNT:

    ret, frame = camera.read()

    if not ret:
        print("Cannot read frame")
        break


    

    faces = app.get(frame)


    # No face
    if len(faces) == 0:

        cv2.putText(
            frame,
            "NO FACE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # More than one face
    elif len(faces) > 1:

        cv2.putText(
            frame,
            "ONLY ONE FACE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # Exactly one face
    else:

        face = faces[0]

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )


        # Draw face box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Get normalized embedding

        embedding = face.normed_embedding


        embeddings.append(
            embedding.copy()
        )


        text = f"Samples: {len(embeddings)}/{SAMPLE_COUNT}"


        cv2.putText(
            frame,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Small delay between samples
        time.sleep(0.15)


    cv2.imshow(
        "Face Registration",
        frame
    )


    key = cv2.waitKey(1)

    if key == ord("q"):
        break




camera.release()
cv2.destroyAllWindows()




if len(embeddings) < SAMPLE_COUNT:

    print()
    print("Registration failed")
    print(
        f"Only collected {len(embeddings)} "
        f"of {SAMPLE_COUNT} samples"
    )

    exit()




embeddings = np.array(embeddings)


mean_embedding = np.mean(
    embeddings,
    axis=0
)


# Normalize again

mean_embedding = (
    mean_embedding /
    np.linalg.norm(mean_embedding)
)




np.save(
    EMBEDDING_FILE,
    mean_embedding
)


print()
print("====================================")
print(" Registration completed")
print("====================================")
print(f"Saved to: {EMBEDDING_FILE}")
print("Embedding shape:", mean_embedding.shape)