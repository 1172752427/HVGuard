# HVGuard

![Framework](framework_img.png)

HVGuard is a multimodal content safety framework that leverages **text (BERT)**, **vision (ViT)**, and **audio (Wav2Vec)** embeddings, along with reasoning from large language models (LLMs). The extracted multimodal features are cached and reused during training, enabling efficient experimentation and reproducibility.  

---

## 📂 Project Structure

- `embeddings/` – Cached multimodal embeddings (`.pth` files). No additional feature extraction is required during training.  
- `datasets/` – Original dataset contents.  
  - `annotation(new).json`: Cleaned and re-transcribed annotations.  
- `HVGuard.py` – Main training and inference script.  
- `config.jsonl` – Configuration file for experiments.  

---

## 🚀 Reproduction Guide

### Step 1. Install dependencies
Make sure you have Python ≥3.8 and install the required packages:
```bash
pip install -r requirements.txt
```

### Step 2. Run training / prediction
To start training on a dataset:
```bash
python HVGuard.py --dataset_name Multihateclip --language Chinese --num_classes 2 --mode train
```
Available command-line arguments:
```bash
parser.add_argument('--dataset_name', type=str, default='Multihateclip', 
    choices=['Multihateclip', 'HateMM'], help='Dataset name')

parser.add_argument('--language', type=str, default='English', 
    choices=['Chinese', 'English'], help='Language of the dataset')

parser.add_argument('--num_classes', type=int, default=2, 
    choices=[2, 3], help='Number of classes for classification')

parser.add_argument('--mode', type=str, default='predict', 
    choices=['train', 'predict'], help='Training mode or prediction mode')
```
- Use `--mode predict` to directly reproduce results from cached embeddings.  
- To test with multiple random seeds, modify `random_state` inside `HVGuard.py`.  
