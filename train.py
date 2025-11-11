from torch.utils.data import DataLoader
from ner_model.data_utils import CoNLLDataset
from ner_model.model import NERModel
from ner_model.model_attention import NERModelAttention
from ner_model.config import Config
from ner_model.data_utils import collate_fn,collate_fn_new,collate_fn_roberta
import torch
from StrongData import augment_dataset
from transformers import get_linear_schedule_with_warmup
def main():
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载配置
    config = Config()

    # 构建模型并移动到 GPU
    model = NERModelAttention(config).to(device)

    # 加载数据集
    print("Training data path:", config.filename_train)
    print("Development data path:", config.filename_dev)

    # 创建数据集实例
    train_dataset = CoNLLDataset(config.filename_train)
    dev_dataset = CoNLLDataset(config.filename_dev)
    # train_dataset.data = augment_dataset(train_dataset) 无需数据增强
    # print(f"Dataset augmented: {len(train_dataset)} samples")
    # 创建 DataLoader，使用自定义的 collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_new(batch, model.tokenizer, config.tag2idx, config.max_length, device)
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_new(batch, model.tokenizer, config.tag2idx, config.max_length, device)
    )

     # 添加学习率调度器
    total_steps = len(train_loader) * config.nepochs
    warmup_steps = int(0.1 * total_steps)  # 10% 的步骤用于预热
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 设置优化器
    model.optimizer = optimizer
    
    # 训练模型
    model.train_model(train_loader, dev_loader, device, scheduler)
    model.save_session()

if __name__ == "__main__":
    main()