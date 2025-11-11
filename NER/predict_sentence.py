import torch
from transformers import BertTokenizerFast  # 使用快速版本的分词器
from ner_model.model import NERModel
from ner_model.config import Config


def read_text_from_file(file_path):
    """
    从 .txt 文件中读取文本内容并返回字符串。

    Args:
        file_path: .txt 文件的路径。

    Returns:
        text: 文件中的文本内容（字符串）。
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def preprocess_text(text, tokenizer, max_length):
    """
    对输入文本进行预处理，转换为模型输入格式。

    Args:
        text: 输入文本（字符串）。
        tokenizer: 分词器。
        max_length: 最大序列长度。

    Returns:
        input_ids: 分词后的输入 ID。
        attention_mask: 注意力掩码。
        tokens: 分词后的单词列表。
    """
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,  # 获取分词后的偏移量
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])  # 将输入 ID 转换为单词
    return encoding["input_ids"], encoding["attention_mask"], tokens


def predict_text(model, tokenizer, text, max_length, idx_to_tag):
    """
    使用训练好的模型对输入文本进行预测。

    Args:
        model: 训练好的 NER 模型。
        tokenizer: 分词器。
        text: 输入文本（字符串）。
        max_length: 最大序列长度。
        idx_to_tag: 将索引映射到标签的字典。

    Returns:
        word_label_pairs: 单词和标签的对应列表。
    """
    # 预处理文本
    input_ids, attention_mask, tokens = preprocess_text(text, tokenizer, max_length)

    # 将输入数据移动到模型所在的设备
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # 模型预测
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    # 将预测结果转换为标签
    predicted_tags = [idx_to_tag[idx] for idx in preds[0]]

    # 过滤掉填充部分的标签和单词
    word_label_pairs = [
        (token, tag)
        for token, tag, mask in zip(tokens, predicted_tags, attention_mask[0])
        if mask == 1 and token not in ["[CLS]", "[SEP]", "[PAD]"]  # 过滤特殊标记
    ]

    return word_label_pairs


def predict_text_1(model, tokenizer, text, max_length, idx_to_tag):
    input_ids, attention_mask, tokens = preprocess_text(text, tokenizer, max_length)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    predicted_tags = [idx_to_tag[idx] for idx in preds[0]]
    entities = []
    current_entity = []
    for token, tag, mask in zip(tokens, predicted_tags, attention_mask[0]):
        if mask == 1 and token not in ["[CLS]", "[SEP]", "[PAD]"]:
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"text": [token], "type": tag.split("-")[1]}
            elif tag.startswith("I-"):
                if current_entity and current_entity["type"] == tag.split("-")[1]:
                    current_entity["text"].append(token)
                else:
                    entities.append(current_entity)
                    current_entity = []
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = []
    return entities


def main():
    # 加载配置
    config = Config()

    # 构建模型
    model = NERModel(config)

    # 加载模型权重
    model.restore_session(config.dir_model)
    model.to(model.device)
    # 加载分词器
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_model_name)  # 使用快速版本的分词器

    # 将索引映射到标签
    idx_to_tag = {idx: tag for tag, idx in config.tag2idx.items()}

    # 从 .txt 文件中读取文本
    file_path = 'D:\\PythonProject\\LiResolver_copy\\EE5\\Apache-2.0.txt'
    text = read_text_from_file(file_path)
    # text = 'License hereby grants to any person obtaining a copy of the Original Work'
    # 进行预测
    word_label_pairs = predict_text_1(model, tokenizer, text, config.max_length, idx_to_tag)

    # 逐个输出单词和标签
    print("Input Text:", text)
    print("Word-Label Pairs:",word_label_pairs)
    #for word, label in word_label_pairs:
        #print(f"{word}: {label}")


if __name__ == "__main__":
    main()