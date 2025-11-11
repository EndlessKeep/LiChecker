import torch
from torch.utils.data import DataLoader
from ner_model.data_utils import CoNLLDataset, collate_fn_new
from ner_model.model import NERModel
from ner_model.config import Config
from sklearn.metrics import classification_report


def predict(model, data_loader):
    """使用训练好的模型进行预测"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            preds = model.predict(input_ids, attention_mask)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    return all_preds, all_labels


def get_tag_metrics(predictions, true_labels, idx_to_tag):
    """计算每个标签的统计指标"""
    tags = list(idx_to_tag.values())
    return classification_report(
        true_labels,
        predictions,
        labels=list(idx_to_tag.keys()),
        target_names=tags,
        zero_division=0,
        output_dict=True
    )


def main():
    # 加载配置
    config = Config()

    # 构建模型
    model = NERModel(config)
    model.restore_session(config.dir_model)
    model.to(model.device)

    # 加载测试数据集
    test_dataset = CoNLLDataset(config.filename_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_new(batch, model.tokenizer, config.tag2idx, config.max_length, model.device)
    )

    # 预测
    predictions, true_labels = predict(model, test_loader)

    # 过滤无效标签 (-100)
    filtered_preds = []
    filtered_labels = []
    for pred, label in zip(predictions, true_labels):
        if label != -100:  # 忽略填充标签
            filtered_preds.append(pred)
            filtered_labels.append(label)

    # 获取每个标签的 ACC、P、R、F1
    idx_to_tag = {idx: tag for tag, idx in config.tag2idx.items()}
    report = get_tag_metrics(filtered_labels, filtered_preds, idx_to_tag)

    # 打印每个标签的精确率（P）、召回率（R）、F1（F1-score）、支持数（Support）
    print("\nDetailed Report per Tag:")
    print(f"{'Tag':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-score':>10} | {'Support':>10}")
    print("-" * 65)
    for tag in sorted(report.keys()):
        if tag not in ["accuracy", "macro avg", "weighted avg"]:
            stats = report[tag]
            print(
                f"{tag:<15} | {stats['precision']:>10.2f} | "
                f"{stats['recall']:>10.2f} | {stats['f1-score']:>10.2f} | "
                f"{stats['support']:>10}"
            )

    # 打印宏观平均值和加权平均值
    for metric in ["macro avg", "weighted avg"]:
        if metric in report:
            stats = report[metric]
            print("-" * 65)
            print(
                f"{metric:<15} | {stats['precision']:>10.2f} | "
                f"{stats['recall']:>10.2f} | {stats['f1-score']:>10.2f} | "
                f"{stats['support']:>10}"
            )

    # 打印总体 ACC
    print("\nOverall Accuracy:", report["accuracy"])


if __name__ == "__main__":
    main()
