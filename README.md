# Deepfake Detection Using Deep Learning

A deep learning project to detect AI-generated (fake) face images using MobileNetV2 transfer learning.

## Team Members
- Darshan Pathak (1AUA23BCS030)
- Neel Khatri (1AUA23BCS122)  
- Nisarg Bhavsar (1AUA23BCS127)

## Faculty Mentor
Ms. Diya Vadhvani | Adani University

## Dataset
Real and Fake Face Detection — Kaggle
https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

## Setup Instructions
1. Download the dataset from Kaggle
2. Install dependencies:
   pip install tensorflow opencv-python scikit-learn matplotlib seaborn pillow tqdm streamlit
3. Update DATASET_PATH and OUTPUT_PATH in Cell 2
4. Run all cells in order
5. Launch app: streamlit run app.py

## Results
- Validation Accuracy: 73.6%
- Validation AUC-ROC: 0.7356
- Model: MobileNetV2 (Transfer Learning)
- Framework: TensorFlow 2.10
