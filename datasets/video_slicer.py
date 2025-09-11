import cv2
import numpy as np
import os
import shutil

# Extract frames from video datasets
# Compute the indices of frames to extract
def slice_frames(video_path, output_dir, num_frames=32):
    print(f"Processing video: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)

    if os.path.exists(video_output_dir):
        print(f"Skipping {video_name}: Frames already extracted.")
        return

    os.makedirs(video_output_dir)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps}")

    if num_frames <= total_frames:
        seg_size = (total_frames - 1) / num_frames
        selected_ids = [int(np.round(seg_size * i)) for i in range(num_frames)]
    else:
        selected_ids = list(range(total_frames)) * \
            (num_frames // total_frames + 1)
        selected_ids = selected_ids[:num_frames]

    print(f"Selected frame indices: {selected_ids}")

    count = 0
    saved_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if count in selected_ids:
            image_name = f"frame_{saved_count + 1:03d}.jpg"
            output_path = os.path.join(video_output_dir, image_name)
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
            saved_count += 1

        count += 1

    cap.release()
    print(f"Finished extracting {saved_count} frames.")

def process_folder(input_folder, output_folder, num_frames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_extensions = {".mp4", ".avi", ".mkv", ".mov"}
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in video_extensions:
            slice_frames(file_path, output_folder, num_frames)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from videos in a folder.")
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the input folder containing videos.")
    parser.add_argument("-o", "--output_folder", type=str,
                        default="frames", help="Path to the output folder.")
    parser.add_argument("--num_frames", type=int, default=2,
                        help="Number of frames to extract per video.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(args.input_folder, args.output_folder, args.num_frames)
