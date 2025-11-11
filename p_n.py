# 这个是输出p和n指标的代码
import torch
from torch.utils.data import DataLoader
from ner_model.model_attention import NERModelAttention
from ner_model.data_utils import CoNLLDataset, collate_fn_new
from ner_model.model import NERModel
from ner_model.config import Config
from sklearn.metrics import classification_report
from tqdm import tqdm
import json
import numpy as np


def predict(model, data_loader):
    """使用训练好的模型进行预测，保持二维结构"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="预测中"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            preds = model.predict(input_ids, attention_mask)
            
            # 保持二维结构
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # 合并批次，保持二维结构
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_preds, all_labels


def get_tag_metrics(predictions, true_labels, idx_to_tag):
    """计算每个标签的统计指标"""
    # 将预测和标签展平为一维数组
    flat_preds = predictions.flatten()
    flat_labels = true_labels.flatten()
    
    # 过滤掉填充标签
    valid_indices = flat_labels != -100
    filtered_preds = flat_preds[valid_indices]
    filtered_labels = flat_labels[valid_indices]
    
    tags = list(idx_to_tag.values())
    return classification_report(
        filtered_labels,
        filtered_preds,
        labels=list(idx_to_tag.keys()),
        target_names=tags,
        zero_division=0,
        output_dict=True
    )


def extract_entities(seq, idx2tag):
    """
    从序列中提取实体，正确处理填充标签和子词对齐
    
    Args:
        seq: 标签序列
        idx2tag: 索引到标签的映射
        
    Returns:
        entities: 提取的实体列表，每个实体为(start, end, type)
    """
    entities = []
    entity_start = None
    entity_type = None
    seq_len = len(seq)
    
    i = 0
    while i < seq_len:
        # 跳过填充标签
        if seq[i] == -100:
            i += 1
            continue
            
        tag = idx2tag.get(seq[i].item(), "O")
        
        if tag == "O" or tag == "[PAD]":
            if entity_start is not None:
                # 结束当前实体
                entities.append((entity_start, i - 1, entity_type))
                entity_start = None
                entity_type = None
        elif tag.startswith("B-"):
            if entity_start is not None:
                # 结束前一个实体
                entities.append((entity_start, i - 1, entity_type))
            
            # 开始新实体
            entity_start = i
            entity_type = tag[2:]  # 去掉"B-"前缀
        elif tag.startswith("I-"):
            current_type = tag[2:]  # 去掉"I-"前缀
            
            # 如果没有前导B-标签，则视为B-标签开始一个新实体
            if entity_start is None:
                entity_start = i
                entity_type = current_type
            # 如果类型不匹配，则结束前一个实体并开始新实体
            elif entity_type != current_type:
                entities.append((entity_start, i - 1, entity_type))
                entity_start = i
                entity_type = current_type
        
        i += 1
    
    # 处理序列结束时的实体
    if entity_start is not None:
        entities.append((entity_start, seq_len - 1, entity_type))
    
    return entities


def compute_entity_metrics(predictions, labels, idx2tag):
    """
    计算实体级别的精确率和召回率，正确处理填充标签
    
    Args:
        predictions: 预测的标签序列 [batch_size, seq_len]
        labels: 真实标签序列 [batch_size, seq_len]
        idx2tag: 索引到标签的映射
        
    Returns:
        metrics: 包含每种实体类型指标的字典
    """
    entity_types = ["Action", "Recipient", "Attitude", "Condition"]
    
    # 初始化计数器
    metrics = {
        entity_type: {"tp": 0, "fp": 0, "fn": 0} 
        for entity_type in entity_types + ["Total"]
    }
    
    # 遍历每个样本
    for i in range(len(predictions)):
        # 提取预测的实体
        pred_entities = extract_entities(predictions[i], idx2tag)
        # 提取真实的实体
        true_entities = extract_entities(labels[i], idx2tag)
        
        # 按实体类型分组
        pred_entities_by_type = {entity_type: [] for entity_type in entity_types}
        true_entities_by_type = {entity_type: [] for entity_type in entity_types}
        
        for entity in pred_entities:
            entity_type = entity[2]
            if entity_type in pred_entities_by_type:
                pred_entities_by_type[entity_type].append(entity)
        
        for entity in true_entities:
            entity_type = entity[2]
            if entity_type in true_entities_by_type:
                true_entities_by_type[entity_type].append(entity)
        
        # 计算每种实体类型的TP、FP、FN
        for entity_type in entity_types:
            pred_entities_of_type = pred_entities_by_type[entity_type]
            true_entities_of_type = true_entities_by_type[entity_type]
            
            # 找到匹配的实体（完全匹配）
            tp = 0
            matched_pred = set()
            matched_true = set()
            
            for p_idx, p_entity in enumerate(pred_entities_of_type):
                for t_idx, t_entity in enumerate(true_entities_of_type):
                    if t_idx in matched_true:
                        continue
                    
                    # 完全匹配：起始位置和结束位置都相同
                    if p_entity[0] == t_entity[0] and p_entity[1] == t_entity[1]:
                        tp += 1
                        matched_pred.add(p_idx)
                        matched_true.add(t_idx)
                        break
            
            # 计算FP和FN
            fp = len(pred_entities_of_type) - tp
            fn = len(true_entities_of_type) - tp
            
            # 更新计数器
            metrics[entity_type]["tp"] += tp
            metrics[entity_type]["fp"] += fp
            metrics[entity_type]["fn"] += fn
            
            # 更新总计数
            metrics["Total"]["tp"] += tp
            metrics["Total"]["fp"] += fp
            metrics["Total"]["fn"] += fn
    
    # 计算精确率和召回率
    for entity_type in metrics:
        tp = metrics[entity_type]["tp"]
        fp = metrics[entity_type]["fp"]
        fn = metrics[entity_type]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[entity_type]["precision"] = precision
        metrics[entity_type]["recall"] = recall
        metrics[entity_type]["f1"] = f1
    
    return metrics


def compute_entity_metrics_relaxed(predictions, labels, idx2tag, iou_threshold=0.5):
    """
    使用松弛匹配计算实体级别的精确率和召回率
    
    Args:
        predictions: 预测的标签序列 [batch_size, seq_len]
        labels: 真实标签序列 [batch_size, seq_len]
        idx2tag: 索引到标签的映射
        iou_threshold: IoU阈值，默认为0.5
        
    Returns:
        metrics: 包含每种实体类型指标的字典
    """
    entity_types = ["Action", "Recipient", "Attitude", "Condition"]
    
    # 初始化计数器
    metrics = {
        entity_type: {"tp": 0, "fp": 0, "fn": 0} 
        for entity_type in entity_types + ["Total"]
    }
    
    # 遍历每个样本
    for i in range(len(predictions)):
        # 提取预测的实体
        pred_entities = extract_entities(predictions[i], idx2tag)
        # 提取真实的实体
        true_entities = extract_entities(labels[i], idx2tag)
        
        # 按实体类型分组
        pred_entities_by_type = {entity_type: [] for entity_type in entity_types}
        true_entities_by_type = {entity_type: [] for entity_type in entity_types}
        
        for entity in pred_entities:
            entity_type = entity[2]
            if entity_type in pred_entities_by_type:
                pred_entities_by_type[entity_type].append(entity)
        
        for entity in true_entities:
            entity_type = entity[2]
            if entity_type in true_entities_by_type:
                true_entities_by_type[entity_type].append(entity)
        
        # 计算每种实体类型的TP、FP、FN
        for entity_type in entity_types:
            pred_entities_of_type = pred_entities_by_type[entity_type]
            true_entities_of_type = true_entities_by_type[entity_type]
            
            # 找到匹配的实体（松弛匹配）
            tp = 0
            matched_pred = set()
            matched_true = set()
            
            for p_idx, p_entity in enumerate(pred_entities_of_type):
                p_start, p_end, p_type = p_entity
                best_match = None
                best_iou = 0
                
                for t_idx, t_entity in enumerate(true_entities_of_type):
                    if t_idx in matched_true:
                        continue
                    
                    t_start, t_end, t_type = t_entity
                    
                    # 计算IoU
                    overlap_start = max(p_start, t_start)
                    overlap_end = min(p_end, t_end)
                    overlap = max(0, overlap_end - overlap_start + 1)
                    
                    if overlap > 0:
                        union = (p_end - p_start + 1) + (t_end - t_start + 1) - overlap
                        iou = overlap / union
                        
                        if iou > iou_threshold and iou > best_iou:
                            best_match = t_idx
                            best_iou = iou
                
                if best_match is not None:
                    tp += 1
                    matched_pred.add(p_idx)
                    matched_true.add(best_match)
            
            # 计算FP和FN
            fp = len(pred_entities_of_type) - tp
            fn = len(true_entities_of_type) - tp
            
            # 更新计数器
            metrics[entity_type]["tp"] += tp
            metrics[entity_type]["fp"] += fp
            metrics[entity_type]["fn"] += fn
            
            # 更新总计数
            metrics["Total"]["tp"] += tp
            metrics["Total"]["fp"] += fp
            metrics["Total"]["fn"] += fn
    
    # 计算精确率和召回率
    for entity_type in metrics:
        tp = metrics[entity_type]["tp"]
        fp = metrics[entity_type]["fp"]
        fn = metrics[entity_type]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[entity_type]["precision"] = precision
        metrics[entity_type]["recall"] = recall
        metrics[entity_type]["f1"] = f1
    
    return metrics


def main():
    # 加载配置
    config = Config()

    # 构建模型
    model = NERModelAttention(config)
    model.restore_session(config.dir_model)
    model.to(model.device)

    # 加载测试数据集
    test_dataset = CoNLLDataset("D:\\PythonProject\\LiResolver_copy\\EE5\\data\\ner_test_new.txt")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_new(batch, model.tokenizer, config.tag2idx, config.max_length, model.device)
    )

    # 收集所有预测和标签
    print("正在进行预测...")
    all_predictions, all_labels = predict(model, test_loader)
    
    # 创建索引到标签的映射
    idx_to_tag = {idx: tag for tag, idx in config.tag2idx.items()}
    
    # 获取标签级别的指标
    print("计算标签级别指标...")
    report = get_tag_metrics(all_predictions, all_labels, idx_to_tag)

    # 打印每个标签的精确率（P）、召回率（R）、F1（F1-score）、支持数（Support）
    print("\n标签级别评估结果:")
    print(f"{'标签':<15} | {'精确率(P%)':<15} | {'召回率(R%)':<15} | {'F1分数':<15} | {'支持数':<10}")
    print("-" * 75)
    for tag in sorted(report.keys()):
        if tag not in ["accuracy", "macro avg", "weighted avg"]:
            stats = report[tag]
            print(
                f"{tag:<15} | {stats['precision']*100:>15.2f} | "
                f"{stats['recall']*100:>15.2f} | {stats['f1-score']*100:>15.2f} | "
                f"{stats['support']:>10}"
            )

    # 打印宏观平均值和加权平均值
    for metric in ["macro avg", "weighted avg"]:
        if metric in report:
            stats = report[metric]
            print("-" * 75)
            print(
                f"{metric:<15} | {stats['precision']*100:>15.2f} | "
                f"{stats['recall']*100:>15.2f} | {stats['f1-score']*100:>15.2f} | "
                f"{stats['support']:>10}"
            )

    # 打印总体 ACC
    print("\n总体准确率:", report["accuracy"] * 100, "%")
    
    '''# 计算实体级别的指标（严格匹配）
    print("\n计算实体级别指标（严格匹配）...")
    entity_metrics = compute_entity_metrics(all_predictions, all_labels, idx_to_tag)
    
    # 打印实体级别的指标
    print("\n实体级别评估结果（严格匹配）:")
    print("=" * 60)
    print(f"{'实体类型':<15}{'精确率(P%)':<15}{'召回率(R%)':<15}{'F1分数':<15}")
    print("-" * 60)
    
    for entity_type in ["Action", "Recipient", "Attitude", "Condition", "Total"]:
        precision = entity_metrics[entity_type]["precision"] * 100
        recall = entity_metrics[entity_type]["recall"] * 100
        f1 = entity_metrics[entity_type]["f1"] * 100
        
        print(f"{entity_type:<15}{precision:<15.2f}{recall:<15.2f}{f1:<15.2f}")
    
    print("=" * 60)
    
    # 打印详细统计信息
    print("\n详细统计（严格匹配）:")
    for entity_type in ["Action", "Recipient", "Attitude", "Condition", "Total"]:
        tp = entity_metrics[entity_type]["tp"]
        fp = entity_metrics[entity_type]["fp"]
        fn = entity_metrics[entity_type]["fn"]
        print(f"{entity_type}: TP={tp}, FP={fp}, FN={fn}, 总预测={tp+fp}, 总真实={tp+fn}")
    
    # 计算实体级别的指标（松弛匹配）
    print("\n计算实体级别指标（松弛匹配，IoU阈值=0.5）...")
    relaxed_metrics = compute_entity_metrics_relaxed(all_predictions, all_labels, idx_to_tag)
    
    # 打印实体级别的指标（松弛匹配）
    print("\n实体级别评估结果（松弛匹配）:")
    print("=" * 60)
    print(f"{'实体类型':<15}{'精确率(P%)':<15}{'召回率(R%)':<15}{'F1分数':<15}")
    print("-" * 60)
    
    for entity_type in ["Action", "Recipient", "Attitude", "Condition", "Total"]:
        precision = relaxed_metrics[entity_type]["precision"] * 100
        recall = relaxed_metrics[entity_type]["recall"] * 100
        f1 = relaxed_metrics[entity_type]["f1"] * 100
        
        print(f"{entity_type:<15}{precision:<15.2f}{recall:<15.2f}{f1:<15.2f}")
    
    print("=" * 60)
    
    # 打印详细统计信息（松弛匹配）
    print("\n详细统计（松弛匹配）:")
    for entity_type in ["Action", "Recipient", "Attitude", "Condition", "Total"]:
        tp = relaxed_metrics[entity_type]["tp"]
        fp = relaxed_metrics[entity_type]["fp"]
        fn = relaxed_metrics[entity_type]["fn"]
        print(f"{entity_type}: TP={tp}, FP={fp}, FN={fn}, 总预测={tp+fp}, 总真实={tp+fn}")
    
    # 保存结果到文件
    results = {
        "token_level": {tag: stats for tag, stats in report.items()},
        "entity_level_strict": entity_metrics,
        "entity_level_relaxed": relaxed_metrics
    }
    
    with open("d:/PythonProject/LiResolver_copy/EE5/evaluation_attention_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("\n评估结果已保存到 evaluation_results.json")'''


if __name__ == "__main__":
    main()