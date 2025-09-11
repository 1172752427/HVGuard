import os
import argparse
from moviepy.editor import VideoFileClip


# Convert MP4 video to WAV audio file
def convert_video_to_audio(input_folder: str, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp4"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(
                output_folder, os.path.splitext(file_name)[0] + ".wav")

            try:
                with VideoFileClip(input_path) as video:
                    audio = video.audio
                    if audio:
                        audio.write_audiofile(output_path)
                        print(f"Success：{output_path}")
            except Exception as e:
                print(f"Fail：{input_path}，Error message：{e}")


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 video to WAV audio file...")
    parser.add_argument("-i", "--input_folder", type=str,
                        required=True, help="Input video folder path")
    parser.add_argument("-o", "--output_folder", type=str,
                        required=True, help="Output audio folder path")

    args = parser.parse_args()

    convert_video_to_audio(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
