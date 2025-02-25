## SqueezeBERT for Text Classification on MNLI (GLUE Benchmark)

This project fine-tunes the SqueezeBERT transformer model for sequence classification using the MNLI dataset from the GLUE benchmark. The MNLI dataset consists of premise-hypothesis pairs where the task is to predict whether the hypothesis entails, contradicts, or is neutral to the premise.

## 🚀 Project Overview

Loads the MNLI dataset using Hugging Face's datasets library.

Tokenizes the dataset using the SqueezeBERT tokenizer.

Fine-tunes SqueezeBERT (squeezebert/squeezebert-uncased) for multi-class classification (entailment, contradiction, neutral).

Uses Hugging Face's Trainer API for training and evaluation.

Evaluates model performance on matched and mismatched validation sets.

Saves the fine-tuned model for future inference.

## 🛠 Installation

Ensure you have Python 3.7+ and install the required dependencies:

pip install transformers datasets sklearn numpy torch

## 📂 Project Structure

│── main.py               # Script for training and evaluating the model
│── README.md             # Project documentation
│── fine_tuned_squeezebert_mnli/ # Directory for saving the fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── vocab.txt

## 📜 How to Run

1️⃣ Train and Evaluate the Model

Run the script to fine-tune SqueezeBERT:

python main.py

2️⃣ Save and Load the Model for Future Use

The fine-tuned model is saved in the fine_tuned_squeezebert_mnli/ directory. You can reload it using:

from transformers import SqueezeBertForSequenceClassification, SqueezeBertTokenizer

model = SqueezeBertForSequenceClassification.from_pretrained("./fine_tuned_squeezebert_mnli")
tokenizer = SqueezeBertTokenizer.from_pretrained("./fine_tuned_squeezebert_mnli")

## 🏆 Model Performance

The model is evaluated using accuracy, precision, recall, and F1-score on the mismatched validation set.

Example evaluation output:

Evaluation results on mismatched validation set: {'accuracy': ..., 'f1': ..., 'precision': ..., 'recall': ...}

## 📌 References

Hugging Face Transformers

GLUE Benchmark

MNLI Dataset

🔧 Author: [Your Name]📧 Contact: your.email@example.com📅 Last Updated: February 2025

