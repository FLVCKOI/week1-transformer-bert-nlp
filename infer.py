# infer.py
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# 加载模型和分词器
model = TFAutoModelForSequenceClassification.from_pretrained("output/")
tokenizer = AutoTokenizer.from_pretrained("output/")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=256)
    logits = model(**inputs, training=False).logits
    pred = tf.argmax(logits, axis=1).numpy()[0]
    label = "positive" if pred == 1 else "negative"
    return label

# 示例
print(predict_sentiment("This movie was awesome!"))
print(predict_sentiment("Terrible experience, not recommended."))
