import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import generate_sample_dashcam_video

if __name__ == "__main__":
    vid_path = generate_sample_dashcam_video(
        output_path="data/samples/sample_dashcam.mp4",
        num_frames=40,
        fps=20,
        source_images_dir="data/samples"
    )
    print(f"Generated sample dashcam video at: {vid_path}")

