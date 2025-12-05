import cv2
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from picamera2 import Picamera2

# ---------- CONSTANTS ----------
GAP_CALIB_FACTOR = 0.957729
MARKER_SIZE_MM   = 100.0

FRAME_SIZE       = (960, 540)
DRAW_DEBUG       = False

FONT             = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE       = 0.55
TEXT_THICKNESS   = 2
TEXT_COLOR       = (0, 255, 0)


@dataclass(slots=True)
class Marker:
    corners: np.ndarray
    center: np.ndarray
    mm_per_px_top: float


class ArucoGapMeasurementSystem:
    def __init__(self):
        self.start_time = datetime.now()  # program timer starts

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        params = cv2.aruco.DetectorParameters()

        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        # Camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": FRAME_SIZE}
        )
        self.picam2.configure(config)

        self.startup_time_ms = None  # stored after first processed frame
        self.first_frame_processed = False


    @staticmethod
    def order_corners(corners):
        pts = np.asarray(corners, dtype=np.float32)
        idx = np.argsort(pts[:, 1])
        top, bottom = pts[idx][:2], pts[idx][2:]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


    @staticmethod
    def compute_side_lengths_and_ratios(corners):
        tl, tr, br, bl = corners
        sides = np.array([tr - tl, br - tr, bl - br, tl - bl], dtype=np.float32)
        px_lengths = np.linalg.norm(sides, axis=1)
        return MARKER_SIZE_MM / px_lengths


    @classmethod
    def build_marker(cls, raw):
        if raw.ndim == 3:
            raw = raw.reshape((4, 2))
        ordered = cls.order_corners(raw)
        center = ordered.mean(axis=0, dtype=np.float32)
        mm_per_px = cls.compute_side_lengths_and_ratios(ordered)
        return Marker(ordered, center, float(mm_per_px[0]))


    @staticmethod
    def select_left_right(markers):
        xs = np.array([m.center[0] for m in markers], dtype=np.float32)
        return markers[int(np.argmin(xs))], markers[int(np.argmax(xs))]


    @staticmethod
    def compute_gap(left, right):
        l_tl, l_tr, l_br, _ = left.corners
        r_tl, _, r_br, r_bl = right.corners

        mid_left = (l_tr + l_br) * 0.5
        mid_right = (r_tl + r_bl) * 0.5

        vectors = np.vstack([
            r_tl - l_tr,
            mid_right - mid_left,
            r_bl - l_br
        ])

        px_dist = np.linalg.norm(vectors, axis=1)
        avg_mm_px = (left.mm_per_px_top + right.mm_per_px_top) * 0.5
        mm_values = (px_dist * avg_mm_px) * GAP_CALIB_FACTOR

        return float(mm_values.mean())


    def run(self):
        self.picam2.start()
        print("System starting... please wait.\n")

        try:
            while True:
                frame_rgb = self.picam2.capture_array()
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                corners, ids, _ = self.detector.detectMarkers(gray)

                if ids is not None and len(ids) >= 2:
                    markers = [self.build_marker(c) for c in corners]
                    left, right = self.select_left_right(markers)
                    gap = self.compute_gap(left, right)

                    cv2.putText(frame, f"GAP: {gap:.3f} mm", (20, 40),
                                FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

                    # ---------- Capture startup runtime ONLY after first good frame ----------
                    if not self.first_frame_processed:
                        self.startup_time_ms = (datetime.now() - self.start_time).total_seconds() * 1000
                        self.first_frame_processed = True

                        print(f"\n🚀 Program Ready. Startup Time: {self.startup_time_ms:.2f} ms\n")

                        # Display startup time on screen for 2 seconds
                        display_until = datetime.now().timestamp() + 2
                        while datetime.now().timestamp() < display_until:
                            temp = frame.copy()
                            cv2.putText(temp, f"Startup Time: {self.startup_time_ms:.2f} ms",
                                        (20, 80), FONT, 0.6, (0,255,255), 2)
                            cv2.imshow("Gap Measurement", temp)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                self.picam2.stop()
                                cv2.destroyAllWindows()
                                return

                cv2.imshow("Gap Measurement", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    ArucoGapMeasurementSystem().run()
