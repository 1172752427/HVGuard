import os
import json
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import csv

# Extract audio transcripts and prosodic emotions (Fun-ASR)
def load_model(model_dir, device="cuda:0"):
    model = AutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        hub="hf",
    )
    return model

# Perform speech recognition
def recognize_speech(model, audio_path, language="auto", use_itn=True, batch_size_s=60, merge_vad=True, merge_length_s=15):
    result = model.generate(
        input=audio_path,
        cache={},
        language=language, 
        use_itn=use_itn,
        batch_size_s=batch_size_s,
        merge_vad=merge_vad,
        merge_length_s=merge_length_s,
    )
    return result

# Extract emotion labels from the recognition results
def extract_emotion(result):
    emotions = []
    for item in result:
        text = item.get('text', '')
        emotion = None
        for token in text.split('<|'):
            if token.endswith('|>') and token.isupper():
                emotion = token[:-2]
                break
        emotions.append(emotion)
    return emotions

def process_audio_folder(model, folder_path, output_json, title_path, label_path, frames_path):
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    results = []

    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as json_file:
            results = json.load(json_file)
    else:
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump([], json_file, ensure_ascii=False, indent=4)

    with open(title_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    
    with open(label_path, 'r', encoding='utf-8') as video_file:
        label_reader = csv.DictReader(video_file, delimiter='\t')
        label_data = {row['Video_ID']: row['Majority_Voting'] for row in label_reader}

        for wav_file in wav_files:
            audio_path = os.path.join(folder_path, wav_file)
            video_id = os.path.splitext(wav_file)[0]
            print(f"Processing {video_id}...")
            res = recognize_speech(model, audio_path)
            emotions = extract_emotion(res)
            transcript = rich_transcription_postprocess(res[0]["text"])
            emotion = emotions[0] if emotions else "Unknown"
            title = json_data[video_id]['title']
            path1 = frames_path + video_id
            label = label_data[video_id]

            results.append({
                "Video_ID": video_id,
                "Title": title,
                "Transcript": transcript,
                "Emotion": emotion,
                "Frames_path": path1,
                "Audio_path": audio_path,
                "Frames_description": "",
                "Text_description": "",
                "Mix_description": "",
                "Label":label
            })

    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract emotion from videos in a folder.")
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the input folder containing videos.")

    return parser.parse_args()

def main():
    model = "FunAudioLLM/SenseVoiceSmall"
    base_path = f'./HateMM' # (Multihateclip, HateMM)
    model = load_model(model)
    target = ["valid", "test", "train"]

    args = parse_args()
    base_path = args.input_folder
    for i in target:
        folder_path = base_path + "audios/" + i
        output_json = base_path + "annotation(new)"  + ".json"
        title_path = base_path + "text.json"
        label_path = base_path + "annotation/" + i + ".tsv"
        frames_path = base_path +  "frames/" + i + "/"
        process_audio_folder(model, folder_path, output_json, title_path, label_path, frames_path)
    print(f"Emotion analysis results saved to {output_json}")

if __name__ == "__main__":
    main()