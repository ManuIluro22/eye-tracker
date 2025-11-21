import mediapipe as mp
import cv2
import json
import queue
import threading
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from utils import _build_model

from utils import get_config, getLeftEye, getRightEye

# Read config.ini file
SETTINGS, COLOURS, EYETRACKER, TF = get_config("src/config.ini")
SCALE_X = 1.25

class Detector:
    def __init__(self, output_size, show_stream=False, show_markers=False, show_output=False):

        print("Starting face detector...")
        self.output_size = output_size
        self.show_stream = show_stream
        self.show_markers = show_markers
        self.show_output = show_output

        self.face_img = np.zeros((output_size, output_size, 3))
        self.face_align_img = np.zeros((output_size, output_size, 3))
        self.l_eye_img = np.zeros((output_size, output_size, 3))
        self.r_eye_img = np.zeros((output_size, output_size, 3))
        self.head_pos = np.ones((output_size, output_size))
        self.head_angle = 0.0

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Create face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Load landmarks from JSON
        with open('src/landmarks.json', 'r') as f:
            landmarks = json.load(f)

        self.LEFT_EYE = landmarks["LEFT_EYE"]
        self.RIGHT_EYE = landmarks["RIGHT_EYE"]
        self.LEFT_IRIS = landmarks["LEFT_IRIS"]
        self.RIGHT_IRIS = landmarks["RIGHT_IRIS"]

        # Threaded webcam capture
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def get_frame(self):

        frame = self.q.get()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Get feature locations
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            l_eye_pts = mesh_points[self.RIGHT_EYE]
            r_eye_pts = mesh_points[self.LEFT_EYE]
            l_iris_pts = mesh_points[self.LEFT_IRIS]
            r_iris_pts = mesh_points[self.RIGHT_IRIS]

            # Calculate eye centers and head angle
            l_eye_center = l_eye_pts.mean(axis=0).astype("int")
            r_eye_center = r_eye_pts.mean(axis=0).astype("int")
            l_iris_center = l_iris_pts.mean(axis=0).astype("int")
            r_iris_center = r_iris_pts.mean(axis=0).astype("int")
            eye_dist = np.linalg.norm(r_eye_center - l_eye_center)
            dY = r_eye_center[1] - l_eye_center[1]
            dX = r_eye_center[0] - l_eye_center[0]
            self.head_angle = np.degrees(np.arctan2(dY, dX))

            if self.show_markers:
                for point in l_eye_pts:
                    cv2.circle(frame, (point[0], point[1]), 1, COLOURS["blue"], -1)

                for point in r_eye_pts:
                    cv2.circle(frame, (point[0], point[1]), 1, COLOURS["blue"], -1)

                # iris
                cv2.circle(frame, (l_iris_center[0], l_iris_center[1]), 3, COLOURS["green"], 1)
                cv2.circle(frame, (r_iris_center[0], r_iris_center[1]), 3, COLOURS["green"], 1)

            # Face extraction and alignment
            desired_l_eye_pos = (0.35, 0.5)
            desired_r_eye_posx = 1.0 - desired_l_eye_pos[0]

            desired_dist = desired_r_eye_posx - desired_l_eye_pos[0]
            desired_dist *= self.output_size
            scale = desired_dist / eye_dist

            eyeCenter = (
                (l_eye_center[0] + r_eye_center[0]) / 2,
                (l_eye_center[1] + r_eye_center[1]) / 2,
            )

            t_x = self.output_size * 0.5
            t_y = self.output_size * desired_l_eye_pos[1]

            align_angles = (0, self.head_angle)
            for angle in align_angles:
                M = cv2.getRotationMatrix2D(eyeCenter, angle, scale)
                M[0, 2] += t_x - eyeCenter[0]
                M[1, 2] += t_y - eyeCenter[1]

                aligned = cv2.warpAffine(
                    frame,
                    M,
                    (self.output_size, self.output_size),
                    flags=cv2.INTER_LINEAR,
                )

                if angle == 0:
                    self.face_img = aligned
                else:
                    self.face_align_img = aligned

            try:
                self.l_eye_img = getLeftEye(frame, landmarks, l_eye_center)
                self.l_eye_img = cv2.resize(
                    self.l_eye_img, (self.output_size, self.output_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                self.r_eye_img = getRightEye(frame, landmarks, r_eye_center)
                self.r_eye_img = cv2.resize(
                    self.r_eye_img, (self.output_size, self.output_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            except:
                pass

            # Get position of head in the frame
            frame_bw = np.ones((frame.shape[0], frame.shape[1])) * 255

            # Create a rectangle around the face
            for face_landmarks in results.multi_face_landmarks:
                h, w, c = frame.shape
                cx_min = w
                cy_min = h
                cx_max = cy_max = 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx < cx_min:
                        cx_min = cx
                    if cy < cy_min:
                        cy_min = cy
                    if cx > cx_max:
                        cx_max = cx
                    if cy > cy_max:
                        cy_max = cy
                # Draw black rect
                cv2.rectangle(frame_bw, (cx_min, cy_min), (cx_max, cy_max), 0, -1)

            self.head_pos = cv2.resize(frame_bw, (self.output_size, self.output_size),
                                       interpolation=cv2.INTER_LINEAR)

            if self.show_output:
                cv2.imshow("Head position", self.head_pos)
                cv2.imshow(
                    "Face and eyes",
                    np.vstack(
                        (
                            np.hstack((self.face_img, self.face_align_img)),
                            np.hstack((self.l_eye_img, self.r_eye_img)),
                        )
                    ),
                )

            if self.show_stream:
                cv2.imshow("Webcam", frame)

        return self.l_eye_img, self.r_eye_img, self.face_img, self.face_align_img, self.head_pos, self.head_angle

    def close(self):
        print("Closing face detector...")
        self.capture.release()
        cv2.destroyAllWindows()


class Predictor:
    def __init__(self,
                 model_cls,
                 model_path: str | Path,
                 cfg_json: str | Path | None = None,
                 gpu: int = 0):

        self.device = (
            torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available()
            else torch.device("cpu")
        )
        model_path = Path(model_path)

        if model_path.suffix == ".ckpt":
            self.model = model_cls.load_from_checkpoint(model_path).to(self.device)
        else:
            if cfg_json is None:
                raise ValueError("cfg_json must be provided for .pt weights")
            cfg = json.loads(Path(cfg_json).read_text())

            # build architecture Single/Eyes/Full
            img_types = cfg.get("img_types",
                                ["face_aligned","l_eye","r_eye","head_pos","head_angle"])
            self.model = _build_model(cfg, img_types).to(self.device)

            state = torch.load(model_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state, strict=True)

        self.model.eval().half()
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _bgr2rgb(img: np.ndarray) -> np.ndarray:
        """OpenCV → PIL colour order with positive strides."""
        if img.ndim == 3 and img.shape[2] == 3:
            return img[..., ::-1].copy()  # ← .copy() removes neg-stride
        return img

    def _prep_imgs(self, imgs):
        tensors = []
        for idx, im in enumerate(imgs):
            im = self._bgr2rgb(im)

            # head-pos mask → 1-channel
            if idx == 3:  # adjust if ordering differs
                if im.ndim == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            if im.dtype != np.uint8:
                im = im.astype(np.uint8)

            tensors.append(
                self.to_tensor(im).unsqueeze(0).half().to(self.device)
            )
        return tensors

    def predict(self, *imgs, head_angle: float | None = None):
        tensors = self._prep_imgs(imgs)
        if head_angle is not None:
            tensors.append(
                torch.tensor([head_angle], dtype=torch.float16, device=self.device)
            )
        with torch.no_grad():
            xy = self.model(*tensors)[0].float().cpu().numpy()
            xy[0] = xy[0] * SCALE_X

        return float(xy[0]), float(xy[1])


if __name__ == "__main__":
    detector = Detector(
        output_size=512, show_stream=False, show_output=True, show_markers=False
    )

    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
            break
        detector.get_frame()

    detector.close()