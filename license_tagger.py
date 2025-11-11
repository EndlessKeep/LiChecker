import torch
import argparse
from ner_model.config import Config
from ner_model.model import NERModel  # 导入标准模型
from ner_model.model_attention import NERModelAttention  # 使用带注意力机制的模型
from transformers import BertTokenizer
import numpy as np

def tokenize_and_predict(text, model, tokenizer, config, device):
    """
    对输入文本进行分词并预测BIO标签
    
    Args:
        text: 输入的许可证文本
        model: 加载好的NER模型
        tokenizer: BERT分词器
        config: 配置对象
        device: 计算设备
    
    Returns:
        tokens: 分词后的文本
        predictions: 预测的BIO标签
    """
    # 分词
    words = text.strip().split()
    
    # 使用BERT分词器处理
    tokens = []
    word_ids = []
    current_word_idx = 0
    
    # 添加[CLS]标记
    tokens.append("[CLS]")
    word_ids.append(None)
    
    # 对每个单词进行分词
    for word in words:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [tokenizer.unk_token]
        
        tokens.extend(word_tokens)
        word_ids.extend([current_word_idx] * len(word_tokens))
        current_word_idx += 1
    
    # 添加[SEP]标记
    tokens.append("[SEP]")
    word_ids.append(None)
    
    # 转换为ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    
    # 确保不超过最大长度
    max_length = config.max_length
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        word_ids = word_ids[:max_length]
    
    # 填充
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    
    # 转换为张量
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]
    
    # 将预测结果映射回原始单词
    word_predictions = []
    prev_word_idx = None
    
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        
        # 只保留每个单词的第一个子词的预测
        if word_idx != prev_word_idx:
            if i < len(predictions):
                pred_id = predictions[i]
                if pred_id in config.idx_to_tag:
                    word_predictions.append(config.idx_to_tag[pred_id])
                else:
                    word_predictions.append("O")
        
        prev_word_idx = word_idx
    
    return words, word_predictions

def format_output(words, tags):
    """格式化输出结果"""
    result = []
    for word, tag in zip(words, tags):
        result.append(f"{word} {tag}")
    return "\n".join(result)

def main():
    parser = argparse.ArgumentParser(description="使用NER模型对许可证文本进行BIO标注")
    parser.add_argument("--input", type=str, default='D:\PythonProject\LiResolver_copy\license_txt\Apache-2.0.txt', help="输入文本文件路径，如果不提供则从标准输入读取")
    parser.add_argument("--output", type=str, default='D:/PythonProject/LiResolver_copy/EE5/tag_apache.txt', help="输出文件路径，如果不提供则输出到标准输出")
    parser.add_argument("--model_dir", type=str, default=None, help="模型保存目录，默认使用配置中的目录")
    parser.add_argument("--use_attention", action="store_true", help="使用注意力机制模型")
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 设置模型目录
    
    # 设置注意力机制
    if args.use_attention:
        config.use_attention = True
        config.num_attention_heads = 4
        config.attention_dropout = 0.1
        print("使用带有注意力机制的模型进行预测...")
    else:
        config.use_attention = False
        print("使用标准模型进行预测...")
    
    # 创建标签索引到标签名称的映射
    config.idx_to_tag = {idx: tag for tag, idx in config.vocab_tags.items()}
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模

    model = NERModel(config)
    model.restore_session(config.dir_model)
    model.to(device)
    print(f"模型已从 {config.dir_model} 加载")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    
    # 读取输入文本
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        print("请输入许可证文本（输入完成后按Ctrl+D或Ctrl+Z结束）:")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        text = " ".join(lines)
    
    # 预测
    words, predictions = tokenize_and_predict(text, model, tokenizer, config, device)
    
    # 格式化输出
    output = format_output(words, predictions)
    
    # 输出结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"结果已保存到 {args.output}")
    else:
        print("\n预测结果:")
        print("=" * 50)
        print(output)
        print("=" * 50)
    
    # 统计各类实体数量
    entity_counts = {}
    current_entity = None
    for tag in predictions:
        if tag.startswith("B-"):
            entity_type = tag[2:]
            if entity_type not in entity_counts:
                entity_counts[entity_type] = 0
            entity_counts[entity_type] += 1
    
    print("\n实体统计:")
    for entity_type, count in entity_counts.items():
        print(f"{entity_type}: {count}个")

if __name__ == "__main__":
    main()