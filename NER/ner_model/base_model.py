import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from TorchCRF import CRF

class BaseModel(torch.nn.Module):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """
        Args:
            config: (Config instance) class with hyper-parameters, vocab, and embeddings
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.logger = config.logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=config.dir_output)  # TensorBoard writer
        self.optimizer = None
        self.best_score = 0
        self.epochs_no_improve = 0

    def add_train_op(self, lr_method, lr, clip=-1):
        """
        Defines self.optimizer that performs an update on a batch.

        Args:
            lr_method: (string) optimizer method, e.g., "adam", "sgd", etc.
            lr: (float) learning rate
            clip: (float) gradient clipping value. If < 0, no clipping.
        """
        _lr_m = lr_method.lower()  # lower to make sure

        if _lr_m == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif _lr_m == 'adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=lr)
        elif _lr_m == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        elif _lr_m == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError("Unknown method {}".format(_lr_m))

        if clip > 0:  # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

    def initialize_session(self):
        """Initializes the model and moves it to the appropriate device (CPU/GPU)."""
        self.logger.info("Initializing model session")
        self.to(self.device)

    def save_session(self):
        """Saves the model weights."""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        model_path = os.path.join(self.config.dir_model, "model.pt")
        torch.save(self.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

    def restore_session(self, dir_model):
        """Reloads the model weights.

        Args:
            dir_model: (string) directory containing the model weights.
        """
        model_path = os.path.join(dir_model, "model_619.pt")
        self.load_state_dict(torch.load(model_path, map_location=self.device),strict=False)
        self.logger.info(f"Model loaded from {model_path}")

    def close_session(self):
        """Closes the session (no-op in PyTorch)."""
        pass

    from tqdm import tqdm  # 导入 tqdm

    def train_model(self, train_loader, dev_loader, device, scheduler=None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            dev_loader: 验证数据加载器
            device: 设备（CPU/GPU）
            scheduler: 学习率调度器（可选）
        """
        self.logger.info("Starting training...")
        self.add_train_op(self.config.lr_method, self.config.lr, self.config.clip)
        self.initialize_session()
        
        best_score = 0
        nepoch_no_imprv = 0
        
        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
            
            # 训练一个 epoch
            avg_loss = self.run_epoch(train_loader, epoch, device)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            
            # 在验证集上评估
            self.logger.info("Evaluating on development data")
            metrics = self.run_evaluate(dev_loader)
            
            # 记录指标
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            
            # 打印指标
            msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
            self.logger.info(msg)
            
            # 早停策略
            score = metrics["f1"]
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without improvement".format(
                        nepoch_no_imprv))
                    break
            
            # 使用学习率调度器
            if scheduler is not None:
                scheduler.step()
        
        self.logger.info("Restoring best model weights")
        self.restore_session(self.config.dir_model)

    def evaluate(self, test_loader):
        """
        Evaluates the model on the test set.

        Args:
            test_loader: DataLoader for test data.
        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test_loader)
        msg = " - ".join([f"{k} {v:.4f}" for k, v in metrics.items()])
        self.logger.info(msg)

    def run_epoch(self, data_loader, epoch, device):
        """
        运行一个完整的 epoch，包括训练或验证

        Args:
            data_loader: DataLoader 实例，用于加载数据
            epoch: 当前 epoch 的索引
            device: 设备（CPU 或 GPU）

        Returns:
            avg_loss: 平均损失值（训练模式）
            score: 评估分数（验证模式）
        """
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if self.training:
                # 训练模式
                self.optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = self.compute_loss(outputs, labels, attention_mask)
                loss.backward()
                self.optimizer.step()
            else:
                # 验证模式
                with torch.no_grad():
                    outputs = self(input_ids, attention_mask)
                    loss = self.compute_loss(outputs, labels, attention_mask)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_loss = total_loss / len(data_loader)
        if self.training:
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            return avg_loss
        else:
            score = self.compute_score(all_preds, all_labels)
            self.writer.add_scalar("Score/dev", score, epoch)
            return score

    def compute_loss(self, outputs, labels, attention_mask):
        """
        Computes the loss for a batch.

        Args:
            outputs: Model predictions.
            labels: Ground truth labels.

        Returns:
            Loss value.
            :param attention_mask:
        """
        raise NotImplementedError("Subclasses must implement compute_loss")

    def compute_score(self, preds, labels):
        """
        Computes the evaluation score for a batch.

        Args:
            preds: Model predictions.
            labels: Ground truth labels.

        Returns:
            Evaluation score.
        """
        raise NotImplementedError("Subclasses must implement compute_score")

    def run_evaluate(self, test_loader):
        """
        Evaluates the model on the test set and returns metrics.

        Args:
            test_loader: DataLoader for test data.

        Returns:
            Dictionary of metrics.
        """
        self.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self(input_ids,attention_mask)
                preds = torch.argmax(outputs, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        return self.compute_metrics(all_preds, all_labels)

    def compute_metrics(self, preds, labels):
        """
        Computes evaluation metrics.

        Args:
            preds: Model predictions.
            labels: Ground truth labels.

        Returns:
            Dictionary of metrics.
        """
        raise NotImplementedError("Subclasses must implement compute_metrics")