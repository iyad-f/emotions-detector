from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import fer  # type: ignore # no stubs


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, ClassVar

    import numpy as np


class EmotionDetector:
    compliments: ClassVar[dict[str, str]] = {
        "angry": "Your passion is powerful!",
        "disgust": "Your values are admirable!",
        "fear": "Your courage shines through!",
        "happy": "Your smile lights up the room!",
        "sad": "Your strength is inspiring!",
        "surprise": "Your curiosity is captivating!",
        "neutral": "Your calmness is soothing!",
    }

    def __init__(self, winname: str = "Emotion Detector") -> None:
        self._fer = fer.FER(mtcnn=True)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        )
        self.winname = winname

    def detect_faces(self, frame: cv2.typing.MatLike) -> Sequence[cv2.typing.Rect]:
        gray: cv2.typing.MatLike = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    def detect_emotion(self, face_img: np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]) -> str | None:
        # fer is not typed
        emotion, _ = self._fer.top_emotion(face_img)  # type: ignore
        return emotion

    @staticmethod
    def wrap_text(
        text: str, max_width: int, font: int = cv2.FONT_HERSHEY_SIMPLEX, font_scale: float = 0.8, thickness: int = 2
    ) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current_line = words[0]
        for word in words[1:]:
            test_line = f"{current_line} {word}"
            text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        return lines

    def process_frame(self, frame: cv2.typing.MatLike) -> None:
        faces = self.detect_faces(frame)

        for x, y, w, h in faces:
            face_roi = frame[y : y + h, x : x + w]
            emotion = self.detect_emotion(face_roi)

            if emotion is not None:
                compliment = self.compliments[emotion]
                display_text = f"Emotion: {emotion} | {compliment}"
            else:
                display_text = ''

            padding = 20
            max_width = frame.shape[1] - padding * 2

            lines = self.wrap_text(display_text, max_width)

            line_height = 30

            text_y = y + h + line_height

            for line in lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2

                cv2.putText(
                    frame,
                    line,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                text_y += line_height

    def detect_from_image(self, image_path: str) -> None:
        frame = cv2.imread(image_path)
        self.process_frame(frame)

        cv2.imshow(self.winname, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_video_detection(self) -> None:
        video_capture = cv2.VideoCapture(0)

        print("Press 'q' to quit.")
        while True:
            ret, frame = video_capture.read()
            if ret is False:
                break

            self.process_frame(frame)
            cv2.imshow(self.winname, frame)

            if cv2.waitKey(1) == ord("q"):
                break

            if cv2.getWindowProperty(self.winname, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed.")
                break

        video_capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    # for real time video detection
    detector = EmotionDetector()
    detector.run_video_detection()
    # for image detection
    # detector.detect_from_image('example.jpg')


if __name__ == "__main__":
    main()
