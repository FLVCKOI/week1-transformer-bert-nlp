# train.py
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("imdb")
dataset = dataset.shuffle(seed=42)

# 分词器与预处理
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 转换为 TensorFlow Dataset
train_dataset = tokenized_datasets["train"].shuffle(10000).select(range(5000)).to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["label"],
    shuffle=True,
    batch_size=16
)

val_dataset = tokenized_datasets["test"].select(range(1000)).to_tf_dataset(
    columns=["attention_mask", "input_ids"],
    label_cols=["label"],
    shuffle=False,
    batch_size=16
)

# 加载模型
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 编译与训练
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=2)

# 保存模型与分词器
model.save_pretrained("output/")
tokenizer.save_pretrained("output/")
