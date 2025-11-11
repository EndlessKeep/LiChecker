'''
[许可证条款抽取]の基于PyTorch+BERT+Attention模型的评测脚本
'''
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm

from ner_model.data_utils import CoNLLDataset, collate_fn_new
from ner_model.model_attention import NERModelAttention
from ner_model.config import Config


def align_data(data):
    """将词和标签对齐显示

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # 对每个条目创建对齐字符串
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def interactive_shell(model, tokenizer, config, device):
    """创建交互式shell来测试模型

    Args:
        model: NERModelAttention实例
        tokenizer: BERT分词器
        config: 配置对象
        device: 计算设备
    """
    print("""
这是交互模式。
输入'exit'退出。
你可以输入一个句子，例如：
input> I love Paris""")

    while True:
        try:
            sentence = input("input> ")
        except EOFError:
            break

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        # 处理输入文本
        tokens = []
        word_ids = []
        current_word_idx = 0
        
        # 添加[CLS]标记
        tokens.append("[CLS]")
        word_ids.append(None)
        
        # 对每个单词进行分词
        for word in words_raw:
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
                    if pred_id in model.idx_to_tag:
                        word_predictions.append(model.idx_to_tag[pred_id])
                    else:
                        word_predictions.append("O")
            
            prev_word_idx = word_idx
        
        # 显示结果
        to_print = align_data({"input": words_raw, "output": word_predictions})
        for key, seq in to_print.items():
            print(f"{key}: {seq}")


def get_chunks(labs):
    '''
    从BIO标签序列中提取实体块
    
    Args:
        labs: [O,O,O,O,B-X,I-X,I-X,O,O,O...]
    
    Returns:
        实体列表 [tag,start,end]，左闭右开
        [["X", 0, 2], ["Y", 3, 4]...]
    '''
    chunks = []
    current_chunk = []
    
    for i, tag in enumerate(labs):
        if tag == 'O':
            if current_chunk:
                chunks.append(tuple(current_chunk))
                current_chunk = []
        elif tag.startswith('B-'):
            if current_chunk:
                chunks.append(tuple(current_chunk))
            current_chunk = [tag[2:], i, i+1]
        elif tag.startswith('I-'):
            if current_chunk and current_chunk[0] == tag[2:]:
                current_chunk[2] = i+1
            else:
                # 处理错误的I-标签（没有前导B-标签）
                if current_chunk:
                    chunks.append(tuple(current_chunk))
                current_chunk = [tag[2:], i, i+1]  # 将其视为B-标签
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(tuple(current_chunk))
    
    return chunks


def get_intersection_set(labs1, labs2):
    '''
    计算两个实体集合的交集
    
    Args:
        labs1: 第一个实体集合
        labs2: 第二个实体集合
    
    Returns:
        交集列表
    '''
    intersection = []
    for la1 in labs1:
        found = False
        for la2 in labs2:
            if la1[0] == la2[0]:  # 相同类型的实体
                # 检查是否有重叠
                for i in range(la1[1], la1[2]):
                    if i >= la2[1] and i < la2[2]:
                        intersection.append(la1)
                        found = True
                        break
            if found:
                break
    return intersection


def predict_file(model, tokenizer, config, file_path, output_path, device):
    """
    对单个文件进行预测并保存结果
    
    Args:
        model: 模型实例
        tokenizer: 分词器
        config: 配置对象
        file_path: 输入文件路径
        output_path: 输出文件路径
        device: 计算设备
    """
    # 读取文件
    words = []
    tags = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                if len(parts) >= 2:
                    words.append(parts[0])
                    tags.append(parts[1])
    
    # 分批处理文本
    batch_size = 32
    predictions = []
    
    for i in range(0, len(words), batch_size):
        batch_words = words[i:i+batch_size]
        
        # 处理当前批次
        tokens = ["[CLS]"] + batch_words + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # 填充到最大长度
        padding_length = config.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
        else:
            input_ids = input_ids[:config.max_length]
            attention_mask = attention_mask[:config.max_length]
        
        # 转换为张量
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            batch_preds = torch.argmax(logits, dim=2).cpu().numpy()[0]
        
        # 提取有效预测（去除[CLS]和[SEP]）
        valid_preds = batch_preds[1:len(batch_words)+1]
        
        # 将预测ID转换为标签
        pred_tags = [model.idx_to_tag.get(p, "O") for p in valid_preds]
        predictions.extend(pred_tags)
    
    # 保存预测结果
    with open(output_path, 'w', encoding="utf-8") as f:
        for word, pred in zip(words, predictions):
            f.write(f"{word} {pred}\n")


def print_predictions(config, model, tokenizer, device):
    """
    对测试目录中的所有文件进行预测并保存结果
    
    Args:
        config: 配置对象
        model: 模型实例
        tokenizer: 分词器
        device: 计算设备
    """
    print("phase 1 - LicenseTerm predict --------------------------------------- ")
    
    # 确保输出目录存在
    if not os.path.exists(config.filename_dir_pre):
        os.makedirs(config.filename_dir_pre)
    
    # 遍历测试目录中的所有文件
    file_count = 0
    for root, dirs, files in os.walk(config.filename_dir_test):
        total_files = len(files)
        for file in tqdm(files, desc="处理文件"):
            file_count += 1
            print(f"{file_count}/{total_files} : {file}")
            
            input_path = os.path.join(config.filename_dir_test, file)
            output_path = os.path.join(config.filename_dir_pre, file)
            
            predict_file(model, tokenizer, config, input_path, output_path, device)
    
    print("phase 1 - LicenseTerm predict ----- FINISH ! ")


def evaluate(config, device):
    """
    评估模型在测试集上的性能
    
    Args:
        config: 配置对象
        device: 计算设备
    
    Returns:
        metrics: 评估指标字典
    """
    print("--------------------------------------- ")
    
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.
    
    for root, dirs, files in os.walk(config.filename_dir_test):
        for file in tqdm(files, desc="评估文件"):
            # 读取真实标签
            words = []
            true_tags = []
            with open(os.path.join(config.filename_dir_test, file), 'r', encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            words.append(parts[0])
                            true_tags.append(parts[1])
            
            # 读取预测标签
            pred_tags = []
            with open(os.path.join(config.filename_dir_pre, file), 'r', encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            pred_tags.append(parts[1])
            
            # 确保标签长度一致
            if len(true_tags) != len(pred_tags):
                print(f"警告：文件 {file} 中的真实标签和预测标签长度不一致")
                min_len = min(len(true_tags), len(pred_tags))
                true_tags = true_tags[:min_len]
                pred_tags = pred_tags[:min_len]
            
            # 计算标签级别的准确率
            accs.extend([a == b for a, b in zip(true_tags, pred_tags)])
            
            # 计算实体级别的指标
            true_chunks = set(get_chunks(true_tags))
            pred_chunks = set(get_chunks(pred_tags))
            
            # 计算交集（正确预测的实体）
            correct_entities = get_intersection_set(pred_chunks, true_chunks)
            
            correct_preds += len(correct_entities)
            total_preds += len(pred_chunks)
            total_correct += len(true_chunks)
    
    # 计算精确率、召回率和F1
    p = correct_preds / total_preds if total_preds > 0 else 0
    r = correct_preds / total_correct if total_correct > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    acc = np.mean(accs)
    
    print("正确预测的实体数: " + str(correct_preds))
    print("预测的实体总数: " + str(total_preds))
    print("真实的实体总数: " + str(total_correct))
    
    return {"acc": 100 * acc, "p": 100 * p, "r": 100 * r, "f1": 100 * f1}


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估NER模型')
    parser.add_argument('--interactive', action='store_true', help='启用交互模式')
    parser.add_argument('--predict', action='store_true', help='对测试集进行预测')
    parser.add_argument('--evaluate', action='store_true', help='评估模型性能')
    parser.add_argument('--model_dir', type=str, default=None, help='模型目录')
    parser.add_argument('--use_attention', action='store_true', help='使用注意力机制')
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认执行所有操作
    if not (args.interactive or args.predict or args.evaluate):
        args.interactive = True
        args.predict = True
        args.evaluate = True
    
    # 加载配置
    config = Config()
    
    # 设置注意力机制
    if args.use_attention:
        config.use_attention = True
        config.num_attention_heads = 4
        config.attention_dropout = 0.1
        print("使用带有注意力机制的模型")
    else:
        config.use_attention = False
        print("使用标准模型")
    
    # 设置模型目录
    if args.model_dir:
        config.dir_model = args.model_dir
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    if config.use_attention:
        model = NERModelAttention(config)
    else:
        from ner_model.model import NERModel
        model = NERModel(config)
    
    model.restore_session(config.dir_model)
    model.to(device)
    print(f"模型已从 {config.dir_model} 加载")
    
    # 加载分词器
    tokenizer = model.tokenizer
    
    # 交互模式
    if args.interactive:
        interactive_shell(model, tokenizer, config, device)
    
    # 预测模式
    if args.predict:
        print_predictions(config, model, tokenizer, device)
    
    # 评估模式
    if args.evaluate:
        metrics = evaluate(config, device)
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        print(msg)


if __name__ == "__main__":
    main()