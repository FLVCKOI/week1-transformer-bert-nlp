🎬 BERT for IMDb Sentiment Classification
本项目使用预训练的 BERT 模型对 IMDb 影评数据集进行情感分类（正面 / 负面），涵盖完整的微调、评估和部署流程，并辅以理论笔记与实验分析，适合 NLP 初学者和 Transformers 实践者参考。

📁 项目结构
. ├── train.py # 训练与评估脚本（TensorFlow） ├── infer.py # 推理脚本（输入影评返回预测标签） ├── training_report.md # 分类报告 ├── theory_notes.md # 理论学习整理（Transformer / BERT / Trainer 等） ├── requirements.txt # 所需 Python 库 ├── output/ # 训练后模型及分词器保存位置 └── plots/ # 训练过程 loss / accuracy 可视化图 
🚀 快速开始
安装依赖环境：
pip install -r requirements.txt
训练模型：
python train.py
推理示例：
python infer.py
🔍 示例输出
输入影评： This movie is absolutely wonderful and inspiring.
模型预测： positive

🎓 技术栈
Transformers：BERT-base-uncased（Huggingface Transformers）
数据集：IMDb Large Movie Review Dataset
框架：TensorFlow 2.x + Huggingface 🤗
可视化：Matplotlib, Seaborn
模型保存格式：tf_model.h5 + tokenizer.json
📊 实验结果
Accuracy: 90.3%
Precision / Recall / F1：详见 training_report.md
训练过程图表：

plots/__results___6_0.png（Loss 曲线）
plots/__results___6_1.png（Accuracy 曲线）
plots/__results___8_1.png（Confusion Matrix）
📘 理论整理
Transformer Attention 推导（Q/K/V，多头）
BERT / GPT 结构与训练差异
MLM 与 NSP 原理
Huggingface Trainer 的模块构成
长文本处理方案（如 Longformer / BigBird）
部署建议（FastAPI, 模型压缩优化）
微调技巧（类别不平衡处理、评价指标选择）
🧪 推理代码示例
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification tokenizer = AutoTokenizer.from_pretrained("output/") model = TFAutoModelForSequenceClassification.from_pretrained("output/")
text = "I really enjoyed this movie."
inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
logits = model(**inputs).logits
pred = tf.argmax(logits, axis=1).numpy()[0]
print("Predicted:", "positive" if pred == 1 else "negative")
📦 模型导出文件
output/ 文件夹包含完整模型：

tf_model.h5
tokenizer_config.json
vocab.txt
config.json
📚 推荐资料
Huggingface Transformers 官方文档
Stanford CS224n NLP 课程
Transformers 实战课程
🙋 作者
本项目由厦门大学本科生主导开发，作为从零开始学习大模型与 Transformers 的实战总结。

欢迎联系与交流学习经验、模型部署等！