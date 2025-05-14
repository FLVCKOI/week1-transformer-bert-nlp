🧠 周任务答题整理（理论部分）
一、Transformer 与 BERT/GPT 基础理论
1. Transformer 的基本流程
将输入文本进行 embedding 和 position embedding 相加，之后分别与 Wq、Wk、Wv 矩阵相乘，计算出每个 token 的 Q、K、V 表示。

然后每个 token 的 Q 与所有 token 的 K 做内积，除以 √d_k，再做 softmax 归一化后与 V 相乘加权求和，得到该 token 的输出表示。

2. Attention 的公式流程
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
每个 token 的输出是 softmax(QK^T / √d_k) 后与 V 的加权和。

3. 多头注意力机制 Multi-Head Attention
不是用一组 Q、K、V，而是用 h 组不同的线性变换学习不同的注意力子空间，最后拼接输出。

好处：能关注输入的不同子空间、关系、特征。

4. BERT 与 GPT 的预训练方式有何不同？
BERT：Masked Language Modeling（MLM），输入中随机 mask 一部分 token，模型预测被遮盖的词；使用 Encoder-only 架构。
GPT：Causal Language Modeling（CLM），左到右地预测下一个 token，使用 Decoder-only 架构。
5. BERT 中的 MLM 与 NSP 是什么？
MLM：掩盖部分 token，预测原始 token（上下文都可见）
NSP：Next Sentence Prediction，判断 B 是否为 A 的下一句话，用于学到句子级别关系
6. 预训练与微调的区别？与传统 NLP 有何不同？
预训练：在大语料上无监督或自监督训练语言模型，获得强大语义理解能力。
微调：在下游任务的小数据集上使用少量数据训练以适应具体任务。
优势：预训练模型已具备通用语义能力，微调时无需从零开始；Transformer 架构支持长依赖、并行训练。
7. BERT 文本分类任务大致流程？
加载预训练模型和 tokenizer
对原始文本进行 tokenize，生成输入 ID、attention mask
构建 Dataset，传入 Trainer
设置训练参数（batch size、lr、epoch 等）
训练 + evaluate + 推理
8. 为什么不能用 BERT 来做文本生成任务？
BERT 是双向编码器（encoder-only），输入 token 时可见全部上下文（不可用于自回归生成）。而生成任务通常需要左到右地生成下一个 token，需要 decoder-only 架构。

二、实践能力回顾（IMDb文本分类）
1. truncation=True 的作用
tokenizer 自动截断过长文本至模型支持的最大长度（如 BERT 为 512）。否则输入过长将报错。

2. 是否观察 loss/acc 变化，如何判断过拟合？
观察训练集 loss 降低、准确率升高，同时验证集表现是否同步提升。

若训练 loss 下降但验证 acc 降低或停滞 → 有过拟合
3. Trainer 的构成部分
model：模型结构（AutoModelFor...）
args：TrainingArguments（学习率、batch size 等）
train_dataset, eval_dataset：训练和验证集
tokenizer：用于自动对齐输出
data_collator：批处理方式（如是否使用动态 padding）
compute_metrics：指定评估函数（准确率、F1 等）
4. IMDb 推理阶段完整流程
from transformers import AutoTokenizer, AutoModelForSequenceClassification import torch
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./saved_model")

text = "This movie was surprisingly emotional and beautiful."
inputs = tokenizer(text, return_tensors="pt", truncation=True)
with torch.no_grad():
logits = model(**inputs).logits
pred = torch.argmax(logits, dim=1).item()

print("Prediction:", "positive" if pred == 1 else "negative")
5. 中文文本分类，是否只换 tokenizer 和模型？
还需注意：

使用 AutoTokenizer.from_pretrained("bert-base-chinese")
确保数据是 UTF-8 中文、已清洗（去除异常符号）
标签映射 / 分类数量要匹配
三、拓展与面试风格提问
1. BERT 处理长文本（>512 token）怎么办？
使用专门支持长文本的模型：Longformer、BigBird、LED
将长文切分为段落，逐段输入，最后聚合结果（平均 / attention 加权）
使用句子选择方法，筛选相关度高的 token/句子
2. GPT 更适合实际部署、BERT 适合分类，你同意吗？
不同意。分类任务中 BERT 更轻量、更快推理，部署成本低。GPT 更适合 open-ended 生成式任务。

3. 如何用 BERT 做“评论自动打标签”系统？
构建训练集：用户评论 + 多标签（如"服务好"、"价格高"）
使用多标签分类（sigmoid + binary cross entropy）
注意类别不均衡，可加权 loss
指标选 F1-score、Precision/Recall
4. 模型部署步骤？
保存训练好的模型和 tokenizer（.bin / .pt / .json）
使用 FastAPI、Flask 等部署为接口服务
前端调用 API，进行实时推理
性能优化：ONNX、TorchScript、batch inference、缓存等
5. LoRA 是什么？怎么用于 BERT 微调？
Low-Rank Adaptation（LoRA）是一种高效微调方法，仅训练插入模型中的少量可学习参数（如 query/key/value 中的 rank-1 矩阵），大大减少微调成本。

在 BERT 微调时可以使用 peft 库 (parameter-efficient fine-tuning)，加载 base 模型 + LoRA adapter，仅更新 LoRA 层，适用于低资源任务。

✅ 整理人：@FLVCKOI
📅 时间：2025年5月

回到项目主页：README.md