import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import src.video_processor as video_processor
from src.video_processor import VideoProcessor


class _StubSegmentor:
    def predict(self, image):
        return {"colored_mask": np.zeros_like(image)}


class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.temp_dir.name, "input.avi")
        self.output_path = os.path.join(self.temp_dir.name, "processed.mp4")
        writer = cv2.VideoWriter(
            self.input_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            10,
            (64, 48),
        )
        if not writer.isOpened():
            self.temp_dir.cleanup()
            self.skipTest("OpenCV MJPG writer is unavailable")

        try:
            for index in range(4):
                frame = np.full((48, 64, 3), index * 40, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_video_outputs_decodable_mp4_and_preserves_frame_skip(self):
        processor = VideoProcessor(_StubSegmentor(), frame_skip=2)

        stats = processor.process_video(self.input_path, self.output_path)

        self.assertEqual(stats["total_frames_read"], 4)
        self.assertEqual(stats["frames_inferred"], 2)
        self.assertEqual(stats["frames_skipped"], 2)
        self.assertEqual(stats["output_path"], self.output_path)
        with open(self.output_path, "rb") as output_file:
            # Check MP4 signature (ftyp box)
            header = output_file.read(12)
            self.assertEqual(header[4:8], b"ftyp", "MP4 file should start with ftyp box")

        capture = cv2.VideoCapture(self.output_path)
        self.assertTrue(capture.isOpened())
        decoded_frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded_frames.append(frame)
        finally:
            capture.release()
        self.assertEqual(len(decoded_frames), 4)
        frame_brightness = [int(frame[45, 60].mean()) for frame in decoded_frames]
        self.assertTrue(
            all(
                current > previous + 5
                for previous, current in zip(frame_brightness, frame_brightness[1:])
            ),
            f"Skipped frames should preserve source motion: {frame_brightness}",
        )

    def test_process_video_raises_when_writer_fails(self):
        failed_writer = Mock()
        failed_writer.isOpened.return_value = False
        failed_capture = Mock()
        failed_capture.isOpened.return_value = True
        failed_capture.get.side_effect = [4, 10, 64, 48]
        processor = VideoProcessor(_StubSegmentor(), frame_skip=2)

        with patch("src.video_processor.cv2.VideoCapture", return_value=failed_capture):
            with patch("src.video_processor.cv2.VideoWriter", return_value=failed_writer):
                with self.assertRaisesRegex(RuntimeError, "Failed to initialize video writer"):
                    processor.process_video(self.input_path, self.output_path)

        # writer is set to None when it fails to open, so release is not called
        failed_writer.release.assert_not_called()
        failed_capture.release.assert_called_once()

    def test_processed_video_path_uses_mp4_extension(self):
        path_builder = getattr(video_processor, "processed_video_path", None)
        self.assertTrue(callable(path_builder))

        for input_name in ("drive.mp4", "drive.avi", "drive.mov"):
            with self.subTest(input_name=input_name):
                result = path_builder(input_name, "outputs/predictions")
                # Check that the result ends with .mp4
                self.assertTrue(result.endswith(".mp4"), f"Result {result} should end with .mp4")
                # Check that the result contains the input stem
                self.assertIn("processed_drive", result, f"Result {result} should contain 'processed_drive'")
                # Check that the result contains a UUID-like suffix (8 hex chars before .mp4)
                import re
                uuid_pattern = r"processed_drive_[0-9a-f]{8}\.mp4$"
                self.assertTrue(re.match(uuid_pattern, os.path.basename(result)),
                              f"Result {result} should match pattern {uuid_pattern}")


if __name__ == "__main__":
    unittest.main()
