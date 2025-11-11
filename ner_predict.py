import torch
from torch.utils.data import DataLoader
from ner_model.data_utils import CoNLLDataset, collate_fn,collate_fn_new
from ner_model.model import NERModel
from ner_model.config import Config
from sklearn.metrics import accuracy_score
def predict(model, data_loader):
    """使用训练好的模型进行预测"""
    model.eval()  # 将模型设置为评估模式
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            # 预测
            preds = model.predict(input_ids, attention_mask)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds,all_labels

def main():
    # 加载配置
    config = Config()

    # 构建模型
    model = NERModel(config)

    # 加载模型权重
    model.restore_session(config.dir_model)
    model.to(model.device)
    # 加载测试数据集
    test_dataset = CoNLLDataset(config.filename_test)

    # 创建 DataLoader，使用自定义的 collate_fn
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_new(batch, model.tokenizer, config.tag2idx, config.max_length,device=model.device)
    )

    # 进行预测
    predictions, true_labels = predict(model, test_loader)

    # 将预测结果转换为标签
    idx_to_tag = {idx: tag for tag, idx in config.tag2idx.items()}
    # predicted_tags = [[idx_to_tag[idx] for idx in pred] for pred in predictions]
    # true_tags = [[idx_to_tag[idx] for idx in t if idx != -100] for t in true_labels]

    # 打印预测结果
    #print("Predicted Tags:", predicted_tags)
    #print("True Tags:", true_tags)

    # 过滤 -100 标签并计算 ACC（准确率）
    # 将 predictions 和 true_labels 展平
    flat_preds = []
    flat_labels = []
    for preds_sample, labels_sample in zip(predictions, true_labels):
        for pred, label in zip(preds_sample, labels_sample):
            if label != -100:  # 忽略标签为 -100 的位置
                flat_preds.append(pred)
                flat_labels.append(label)

    # 计算准确率
    acc = accuracy_score(flat_labels, flat_preds)
    print(f"Test Accuracy (ACC): {acc * 100:.2f}%")

    # 打印预测结果
    # for i, tags in enumerate(predicted_tags):
        # print(f"Sample {i + 1}: {tags}")


if __name__ == "__main__":
    main()