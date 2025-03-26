import os
import torch
from transformers import AutoTokenizer, AutoModel


def model_test():
    # 用于测试嵌入模型的样例数据
    sentences = ["样例数据-1", "样例数据-2"]

    # 加载微调后的模型
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'BAAI', 'bge-large-zh-v1.5')
    # model_path = os.path.join(base_dir, 'output', 'matryoshka_nli_BAAI-bge-large-zh-v1.5')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    # 句子分词
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # 短输入到长段落 (short query to long passage, s2p) 检索任务，需添加 instruction
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    # 计算嵌入
    with torch.no_grad():
        model_output = model(**encoded_input)
        # CLS 池化
        sentence_embeddings = model_output[0][:, 0]
    # 归一化嵌入结果
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    print("Sentence embeddings:", sentence_embeddings)

if __name__ == '__main__':
    model_test()
