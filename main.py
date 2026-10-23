import cv2
import numpy as np

from insightface.app import FaceAnalysis

from config import (
    PHONE_CAMERA_URL,
    FACE_MODEL,
    DETECTION_SIZE,
    FACE_THRESHOLD
)

from database import (
    init_db,
    get_all_users
)

from attendance import (
    mark_attendance
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


def recognize():

    init_db()

    users = get_all_users()

    if not users:

        print(
            "No registered users"
        )

        print(
            "Register a user first"
        )

        return


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


    already_marked = set()


    print()
    print("Face recognition started")
    print("Press Q to quit")


    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "Cannot read frame"
            )

            break


        faces = app.get(frame)


        for face in faces:

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )


            current_embedding = (
                face.normed_embedding
            )


            best_user = None

            best_similarity = -1


            # Compare with every registered user

            for user in users:

                similarity = float(
                    np.dot(
                        current_embedding,
                        user["embedding"]
                    )
                )


                if similarity > best_similarity:

                    best_similarity = (
                        similarity
                    )

                    best_user = user


            # =====================================
            # Recognition result
            # =====================================

            if (
                best_user is not None
                and best_similarity >= FACE_THRESHOLD
            ):

                name = best_user["name"]

                user_id = best_user["id"]

                text_color = (
                    0,
                    255,
                    0
                )


                # Mark attendance only once
                # during this recognition session

                if user_id not in already_marked:

                    created = mark_attendance(
                        user_id
                    )

                    if created:

                        print(
                            f"Attendance: "
                            f"{name}"
                        )

                    already_marked.add(
                        user_id
                    )


            else:

                name = "UNKNOWN"

                text_color = (
                    0,
                    0,
                    255
                )


            # Draw box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                text_color,
                2
            )


            # Draw result

            text = (
                f"{name} "
                f"{best_similarity:.2f}"
            )


            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )


        cv2.imshow(
            "FaceID",
            frame
        )


        if cv2.waitKey(1) == ord("q"):

            break


    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    recognize()