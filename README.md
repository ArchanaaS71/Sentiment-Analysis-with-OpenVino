# Sentiment Analysis of IMDb Reviews Using BERT and OpenVINO

## Project Overview
This project implements sentiment analysis on IMDb movie reviews using a fine-tuned BERT model. It includes two implementations: one using standard CPU inference and the other optimized with Intel OpenVINO. The project demonstrates the performance benefits of OpenVINO in terms of inference speed and efficiency.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Inference without OpenVINO](#inference-without-openvino)
- [Inference with OpenVINO](#inference-with-openvino)
- [Comparison](#comparison)
- [Conclusion](#conclusion)

## Installation
Install the necessary packages:
```bash
pip install transformers datasets torch onnx onnxruntime accelerate openvino-dev
```

## Dataset
Load and preprocess the IMDb dataset:
```python
from transformers import BertTokenizer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

## Model Training
Fine-tune the BERT model on the IMDb dataset:
```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./bert_imdb")
tokenizer.save_pretrained("./bert_imdb")
```

## Inference without OpenVINO
Perform inference using the standard CPU approach:
```python
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the trained model
model = BertForSequenceClassification.from_pretrained('./bert_imdb')
tokenizer = BertTokenizer.from_pretrained('./bert_imdb')

# Define a function to perform inference
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()

# Test the model
text = "This movie is fantastic!"
prediction = predict(text)
print("Sentiment:", "Positive" if prediction == 1 else "Negative")
```

## Inference with OpenVINO
Optimize and perform inference using OpenVINO:
```python
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
from openvino.runtime import Core

# Load the trained model
model = BertForSequenceClassification.from_pretrained('./bert_imdb')
tokenizer = BertTokenizer.from_pretrained('./bert_imdb')

# Export the model to ONNX format
dummy_input = torch.randint(0, 100, (1, 512))
torch.onnx.export(model,
                  dummy_input,
                  "bert_sentiment.onnx",
                  input_names=['input_ids'],
                  output_names=['output'],
                  opset_version=14,
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}})

# Load the ONNX model using OpenVINO
ie = Core()
model = ie.read_model(model="bert_sentiment.onnx")
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Define a function to perform inference
def predict(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    result = compiled_model([input_ids])[output_layer]
    return np.argmax(result)

# Test the model
text = "This movie is fantastic!"
prediction = predict(text)
print("Sentiment:", "Positive" if prediction == 1 else "Negative")
```

## Comparison
- **Inference Speed:**
  - Measure and compare the time taken for inference with and without OpenVINO.
- **Efficiency:**
  - Assess the resource utilization and throughput for both approaches.

## Conclusion
This project demonstrates the effectiveness of using Intel's OpenVINO toolkit to optimize BERT model inference for sentiment analysis, providing significant performance improvements over standard CPU deployment. By leveraging hardware acceleration, OpenVINO enables faster and more efficient processing, making it suitable for real-time applications and large-scale data analysis.

There is an additional file which consist of code for sentiment analysis without openvino with just bert model.You can use it further for your reference.
