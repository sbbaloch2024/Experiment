# Import necessary libraries
from datasets import load_dataset
from transformers import SqueezeBertTokenizer, SqueezeBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Step 1: Load the MNLI dataset from GLUE
dataset = load_dataset('glue', 'mnli')

# Step 2: Load the tokenizer for SqueezeBERT
tokenizer = SqueezeBertTokenizer.from_pretrained('squeezebert/squeezebert-uncased')

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=128)

# Apply the tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Load the pre-trained SqueezeBERT model for sequence classification
model = SqueezeBertForSequenceClassification.from_pretrained('squeezebert/squeezebert-uncased', num_labels=3)

# Step 5: Define the evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Step 6: Define the training arguments based on the SqueezeBERT paper hyperparameters
training_args = TrainingArguments(
    output_dir='./results',                          
    eval_strategy="epoch",                           # Replaced evaluation_strategy with eval_strategy
    learning_rate=2e-5,                              
    per_device_train_batch_size=16,                  
    per_device_eval_batch_size=16,                   
    num_train_epochs=5,                              
    weight_decay=0.01,                               
    warmup_steps=int(0.1 * len(tokenized_datasets["train"])),  
)

# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],  # For MNLI matched validation set
    compute_metrics=compute_metrics,
)

# Step 8: Train the model
trainer.train()

# Step 9: Evaluate the model
results = trainer.evaluate(eval_dataset=tokenized_datasets["validation_mismatched"])  # Evaluate on the mismatched validation set
print("Evaluation results on mismatched validation set:", results)

# Optional: Save the fine-tuned model
model.save_pretrained("./fine_tuned_squeezebert_mnli")
tokenizer.save_pretrained("./fine_tuned_squeezebert_mnli")