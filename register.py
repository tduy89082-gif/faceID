import cv2
import numpy as np
import time

from insightface.app import FaceAnalysis

from config import (
    PHONE_CAMERA_URL,
    FACE_MODEL,
    DETECTION_SIZE,
    REGISTRATION_SAMPLES
)

from database import (
    init_db,
    save_user
)


def load_model():

    app = FaceAnalysis(
        name=FACE_MODEL,
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    return app


def register_user():

    init_db()

    name = input(
        "Enter user name: "
    ).strip()

    if not name:

        print("Name cannot be empty")

        return


    print()
    print("Loading FaceID model...")

    app = load_model()

    print("Model loaded")


    camera = cv2.VideoCapture(
        PHONE_CAMERA_URL
    )

    if not camera.isOpened():

        print(
            "Cannot connect to phone camera"
        )

        return


    samples = []

    print()
    print(
        f"Collecting "
        f"{REGISTRATION_SAMPLES} "
        f"face samples"
    )

    print(
        "Look directly at the camera"
    )

    print(
        "Press Q to cancel"
    )


    while len(samples) < REGISTRATION_SAMPLES:

        ret, frame = camera.read()

        if not ret:

            print("Cannot read frame")

            break


        faces = app.get(frame)


        if len(faces) == 0:

            message = "NO FACE"

            color = (
                0,
                0,
                255
            )


        elif len(faces) > 1:

            message = "ONLY ONE FACE"

            color = (
                0,
                0,
                255
            )


        else:

            face = faces[0]

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            embedding = (
                face.normed_embedding
            )

            samples.append(
                embedding.copy()
            )

            message = (
                f"SAMPLES "
                f"{len(samples)}/"
                f"{REGISTRATION_SAMPLES}"
            )

            color = (
                0,
                255,
                0
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            time.sleep(0.1)


        cv2.putText(
            frame,
            message,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


        cv2.imshow(
            "Register User",
            frame
        )


        if cv2.waitKey(1) == ord("q"):

            camera.release()

            cv2.destroyAllWindows()

            print("Registration cancelled")

            return


    camera.release()

    cv2.destroyAllWindows()


    if len(samples) < REGISTRATION_SAMPLES:

        print("Registration failed")

        return


    # Convert to numpy array

    samples = np.asarray(
        samples,
        dtype=np.float32
    )


    # Average embeddings

    embedding = np.mean(
        samples,
        axis=0
    )


    # Normalize

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        print(
            "Invalid embedding"
        )

        return


    embedding = embedding / norm


    user_id = save_user(
        name,
        embedding
    )


    if user_id is None:

        print(
            "This user name already exists"
        )

        return


    print()
    print("==============================")
    print("Registration successful")
    print("==============================")
    print(
        f"User ID: {user_id}"
    )
    print(
        f"Name: {name}"
    )


if __name__ == "__main__":

    register_user()