import torch
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import os
import torch
import json
import torch.nn as nn

from transformers import BertTokenizer, BertModel

# Extract text embeddings(Bert)
text_model = "google-bert/bert-base-uncased"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Text_Model(nn.Module):
    def __init__(self):
        super(Text_Model, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(text_model)
        self.text_model = BertModel.from_pretrained(text_model)
        self.max_length = 128

    def forward(self, text):  # text = f"{title} {transcript}"
        text_input = text
        text_encoding = self.tokenizer(
            text_input,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            text_features = self.text_model(**text_encoding).last_hidden_state
            # Aggregate text features (take the features at the [CLS] position)
            text_features = text_features[:, 0, :].squeeze(0)

        return text_features


def process(json_path):
    model = Text_Model()
    mix_features = {}
    text_features = {}
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        video_id = item.get("Video_ID")
        title = item.get('Title')
        transcript = item.get('Transcript')
        emotion = item.get('emotion')
        Mix_description = item.get('Mix_description')
        Text_description = item.get("Text_description")
        Frames_description = item.get('Frames_description')
        label = item.get('Label')

        text = f"{Mix_description}"
        print(f"Processing {video_id}...")
        out = model(text).to(device)
        text_features[video_id] = out

    return text_features


def save_features_to_pth(features, output_file):
    torch.save(features, output_file)
    print(f"Features saved to {output_file}")


def main():
    json_file = ""
    output_file = ""

    text_features = process(json_file)
    save_features_to_pth(text_features, output_file)


if __name__ == "__main__":
    main()
