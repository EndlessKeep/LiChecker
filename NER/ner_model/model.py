import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import RobertaTokenizer,RobertaModel
from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel
from .config import Config
from TorchCRF import CRF

class NERModel(BaseModel):
    """专门用于命名实体识别（NER）任务的模型类，继承自BaseModel"""

    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含超参数、词汇表等信息
        """
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in self.config.vocab_tags.items()}  # 将标签索引映射到标签名称

        # 定义模型结构
        self.embedding_dim = self.config.hidden_size_lstm
        self.hidden_dim = self.config.hidden_size_lstm
        self.vocab_size = self.config.nwords
        self.tagset_size = self.config.ntags
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        # BERT 嵌入层
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(self.config.bert_model_name)

        # BiLSTM 层
        '''self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )'''

        # 全连接层
        self.hidden2tag = nn.Linear(self.hidden_dim*2, self.tagset_size)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            self.tagset_size
        )

        # Dropout 层
        self.dropout = nn.Dropout(self.config.dropout)

        # CRF 层（如果使用 CRF）
        if self.config.use_crf:
            from TorchCRF import CRF
            self.crf = CRF(self.tagset_size)


    def add_train_op(self, lr_method, lr, clip=-1):
        super().add_train_op(lr_method, lr, clip)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    def forward(self, input_ids, attention_mask):
        """
        前向传播

        Args:
            input_ids: BERT 输入 ID，形状为 (batch_size, seq_len)
            attention_mask: BERT 注意力掩码，形状为 (batch_size, seq_len)

        Returns:
            logits: 模型输出，形状为 (batch_size, seq_len, tagset_size)
        """
        # BERT 嵌入
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        # Dropout
        bert_out = self.dropout(embeddings)
        # 全连接层
        logits = self.classifier(bert_out)
        return logits

    def compute_score(self, preds, labels):
        from sklearn.metrics import f1_score, precision_score, recall_score

        # 过滤掉填充标签（如 -100）
        preds = [p for p, l in zip(preds, labels) if l != -100]
        labels = [l for l in labels if l != -100]

        # 计算 F1、精确率和召回率
        f1 = f1_score(labels, preds, average="macro")
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")

        return {"f1": f1, "precision": precision, "recall": recall}

    def compute_loss(self, logits, labels, attention_mask):
        """
        计算损失

        Args:
            logits: 模型输出，形状为 (batch_size, seq_len, tagset_size)
            labels: 真实标签，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)

        Returns:
            loss: 损失值
        """
        if self.config.use_crf:
            # 处置无效标签（必须替换为有效标签索引）
            pad_label = self.config.tag2idx.get("O")  # 确保此处是"O"的索引
            valid_labels = torch.where(labels == -100,
                                       torch.tensor(pad_label, device=labels.device),
                                       labels)
            logits = logits.transpose(0, 1)  # (seq_len, batch_size, num_tags)
            valid_labels = valid_labels.transpose(0, 1)  # (seq_len, batch_size)
            mask = attention_mask.transpose(0, 1).bool()

            # TorchCRF的损失函数已经是负对数似然
            loss = self.crf(logits, valid_labels, mask=mask).mean()
        else:
            # 交叉熵损失
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充标签
            active_mask = (attention_mask == 1) & (labels != -100)
            active_logits = logits.view(-1, self.tagset_size)[active_mask.view(-1)]
            active_labels = labels.view(-1)[active_mask.view(-1)]
            if active_labels.numel() == 0:
                return torch.tensor(0.0, device=self.device)  # 防止空标签导致的错误
            loss = loss_fn(active_logits, active_labels)
            return loss

    def get_label_acc(self, valid_labels, valid_preds):
        """
        计算每个真实标签的准确率

        Args:
            valid_labels: 有效真实标签（torch.Tensor or numpy.ndarray）
            valid_preds: 有效预测标签（torch.Tensor or numpy.ndarray）
        Returns:
            label_acc: 字典，键为标签名，值为该标签的准确率
        """
        label_acc = {}
        for tag, idx in self.config.vocab_tags.items():
            # 筛选当前标签的所有样本
            mask = (valid_labels == idx)
            if mask.sum() > 0:  # 确保该标签存在
                correct = (valid_preds[mask] == valid_labels[mask]).sum()
                total = mask.sum()
                label_acc[tag] = (correct.item() / total.item()) * 100  # 百分比
        return label_acc

    def predict(self, input_ids, attention_mask):
        """
        预测标签

        Args:
            input_ids: BERT 输入 ID，形状为 (batch_size, seq_len)
            attention_mask: BERT 注意力掩码，形状为 (batch_size, seq_len)

        Returns:
            preds: 预测的标签，形状为 (batch_size, seq_len)
        """
        logits = self.forward(input_ids, attention_mask)
        if self.config.use_crf:
            logits = logits.transpose(0, 1)  # (seq_len, batch_size, num_tags)
            mask = attention_mask.transpose(0, 1)  # (seq_len, batch_size)
            mask = mask.to(self.device)
            preds = self.crf.viterbi_decode(logits, mask=mask.byte())  # 使用 CRF 解码
            # 将预测结果转换回 (batch_size, seq_len)
            preds = [torch.tensor(p).to(self.device) for p in preds]
            preds = torch.nn.utils.rnn.pad_sequence(preds, batch_first=True, padding_value=-100)
        else:
            preds = torch.argmax(logits, dim=-1)
            pad_mask = (attention_mask == 0)
            preds = torch.where(pad_mask, -100, preds)
        return preds

    from tqdm import tqdm  # 导入 tqdm

    def predict1(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        if self.config.use_crf:
            logits = logits.transpose(0, 1)  # (seq_len, batch_size, num_tags)
            mask = attention_mask.transpose(0, 1)  # (seq_len, batch_size)
            mask = mask.to(self.device)
            preds = self.crf.viterbi_decode(logits, mask=mask.bool())  # 使用 bool 类型
            # 将每个 batch 的预测结果填充到 max_length
            max_length = input_ids.size(1)
            padded_preds = []
            for path in preds:
                path_tensor = torch.tensor(path, dtype=torch.long, device=self.device)
                if len(path_tensor) < max_length:
                    padding = torch.full((max_length - len(path_tensor),), -100,
                                         dtype=torch.long, device=self.device)
                    path_tensor = torch.cat([path_tensor, padding])
                padded_preds.append(path_tensor)
            preds = torch.stack(padded_preds)
        else:
            preds = torch.argmax(logits, dim=-1)
        return preds

    def run_epoch(self, data_loader, epoch,device):
        self.train()
        total_loss = 0
        # 使用 tqdm 包装 data_loader
        all_labels = []
        all_preds = []
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{self.config.nepochs}", leave=False)

        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # 前向传播
            logits = self.forward(input_ids, attention_mask)
            loss = self.compute_loss(logits, labels, attention_mask)
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)

            # 过滤无效标签（attention_mask=1 且 labels != -100）
            active_mask = (attention_mask == 1) & (labels != -100)
            valid_labels = labels[active_mask].cpu().numpy()
            valid_preds = preds[active_mask].cpu().numpy()

            all_labels.extend(valid_labels)
            all_preds.extend(valid_preds)

            # 更新进度条描述
            progress_bar.set_postfix({"loss": loss.item(), "acc": (np.mean(valid_preds == valid_labels) * 100) if len(
                valid_labels) > 0 else 0})

        avg_loss = total_loss / len(data_loader)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        overall_acc = np.mean(all_preds == all_labels)
        label_acc = self.get_label_acc(all_labels, all_preds)
        # 打印结果
        print(f"\nEpoch {epoch + 1} Train Loss: {avg_loss:.4f}")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for tag, acc in label_acc.items():
            print(f" - {tag}: {acc:.2f}%")
        return avg_loss

    def get_category_stats(self, valid_labels, valid_preds, valid_mask=None):
        """
        统计四大类别的P/R指标（不区分B/I标签）

        Args:
            valid_labels: 真实标签 [seq_len]
            valid_preds: 预测标签 [seq_len]
            valid_mask: 可选的有效标签掩码

        Returns:
            {
                'Action': {'precision': , 'recall': },
                'Attitude': {'precision': , 'recall': },
                ...
            }
        """
        # 初始化统计字典
        category_stats = {
            'Action': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0},
            'Attitude': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0},
            'Condition': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0},
            'Recipient': {'true_pos': 0, 'false_pos': 0, 'false_neg': 0}
        }

        # 创建标签到类别的映射（不区分B/I）
        tag_to_category = {}
        for tag in self.config.vocab_tags:
            if tag.startswith('B-') or tag.startswith('I-'):
                category = tag.split('-')[1]
                tag_to_category[tag] = category
            else:
                tag_to_category[tag] = None

        # 统计TP/FP/FN
        for true_idx, pred_idx in zip(valid_labels, valid_preds):
            true_tag = self.idx_to_tag.get(true_idx.item(), 'O')
            pred_tag = self.idx_to_tag.get(pred_idx.item(), 'O')

            true_cat = tag_to_category.get(true_tag)
            pred_cat = tag_to_category.get(pred_tag)

            # 统计真实实体（计算Recall）
            if true_cat in category_stats:
                if pred_cat == true_cat:  # True Positive
                    category_stats[true_cat]['true_pos'] += 1
                else:  # False Negative
                    category_stats[true_cat]['false_neg'] += 1

            # 统计预测实体（计算Precision）
            if pred_cat in category_stats:
                if pred_cat != true_cat:  # False Positive
                    category_stats[pred_cat]['false_pos'] += 1

        # 计算各指标
        results = {}
        for category in category_stats:
            tp = category_stats[category]['true_pos']
            fp = category_stats[category]['false_pos']
            fn = category_stats[category]['false_neg']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            results[category] = {
                'precision': precision,
                'recall': recall
            }

        return results

    def run_evaluate(self, test_loader):
        self.eval()
        accs = []
        correct_preds, total_correct, total_preds = 0.0, 0.0, 0.0
        all_valid_labels = []
        all_valid_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # --- 修复1：确保预测与标签形状一致 ---
                preds = self.predict(input_ids, attention_mask)

                # 如果 preds 比 labels 短，填充到相同长度
                if preds.dim() == 1:  # 如果意外被展平
                    preds = preds.view(labels.shape)

                # 确保最终形状一致
                '''if preds.size(0) != labels.size(0):
                    preds = preds[:labels.size(0)]  # 截断多余部分
                if preds.size(1) != labels.size(1):
                    preds = preds[:, :labels.size(1)]'''  # 截断超长序列

                # 生成有效标记
                active_mask = (labels != -100) & (attention_mask == 1)

                # 最终形状验证
                assert preds.shape == labels.shape, \
                    f"Shape mismatch after alignment: preds={preds.shape}, labels={labels.shape}"

                # --- 修复3：安全提取有效部分 ---
                valid_labels = torch.masked_select(labels, active_mask)
                valid_preds = torch.masked_select(preds, active_mask)
                all_valid_labels.append(valid_labels)
                all_valid_preds.append(valid_preds)

                # 累加准确率
                accs.extend((valid_preds == valid_labels).cpu().tolist())

                # 实体级统计
                seq_length = labels.size(1)
                for i in range(labels.size(0)):
                    lab = []
                    pred = []
                    for j in range(seq_length):
                        if active_mask[i, j]:
                            lab_idx = labels[i, j].item()
                            pred_idx = preds[i, j].item()
                            if lab_idx != -100:  # 确认有效标签
                                lab.append(lab_idx)
                                pred.append(pred_idx)
                    # 从数组转为列表，过滤非法实体标签
                    lab = np.array(lab)
                    pred = np.array(pred)
                    lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                    pred_chunks = set(get_chunks(pred, self.config.vocab_tags))
                    correct_preds += len(lab_chunks & pred_chunks)
                    total_correct += len(lab_chunks)
                    total_preds += len(pred_chunks)

        p = correct_preds / total_preds if total_preds > 0 else 0
        r = correct_preds / total_correct if total_correct > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        acc = np.mean(accs) if accs else 0
        '''if len(all_valid_labels) > 0:
            all_valid_labels = torch.cat(all_valid_labels)
            all_valid_preds = torch.cat(all_valid_preds)
            tag_acc = self.get_label_acc(all_valid_labels, all_valid_preds)

            print("\nValidation Per-Tag Accuracy:")
            for tag, acc in sorted(tag_acc.items()):
                print(f"{tag:>15}: {acc:.2f}%")'''
        if len(all_valid_labels) > 0:
            all_valid_labels = torch.cat(all_valid_labels)
            all_valid_preds = torch.cat(all_valid_preds)
            category_stats = self.get_category_stats(all_valid_labels, all_valid_preds)

            print("\nCategory-wise Performance (P/R):")
            for category in ['Action', 'Attitude', 'Condition', 'Recipient']:
                stats = category_stats.get(category, {'precision': 0, 'recall': 0})
                print(f"{category:>10}: P={stats['precision']:.4f}, R={stats['recall']:.4f}")

        return {"acc": 100 * acc, "p": 100 * p, "r": 100 * r, "f1": 100 * f1}