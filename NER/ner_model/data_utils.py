import torch
from transformers import BertTokenizer,RobertaTokenizer,DebertaTokenizer
from transformers import BertModel,RobertaModel,DebertaModel
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import RobertaTokenizer,RobertaModel

class MyIOError(Exception):
    def __init__(self, filename):
        message = """
ERROR: Unable to locate file {}.

FIX: Please ensure the file exists and the path is correct.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(Dataset):
    def __init__(self, filename, processing_word=None, processing_tag=None, max_iter=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.filename, encoding="utf-8") as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        # 确保 words 和 tags 长度一致
                        if len(words) == len(tags):
                            data.append((words, tags))
                        else:
                            print(f"Warning: words and tags length mismatch in sample: {words}, {tags}")
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    if len(ls) >= 2:  # 确保每行至少有两个元素（word 和 tag）
                        word, tag = ls[0], ls[1]
                        if self.processing_word is not None:
                            word = self.processing_word(word)
                        if self.processing_tag is not None:
                            tag = self.processing_tag(tag)
                        words.append(word)
                        tags.append(tag)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BERTProcessor:
    """Class for processing text data using BERT."""

    def __init__(self, model_name='D:\PythonProject\LiResolver_copy\EE5\data/legal-bert-ner', max_length=512):
        """
        Args:
            model_name: name of the BERT model (default: 'bert-base-uncased')
            max_length: maximum sequence length for BERT (default: 512)
        """
        if 'roberta' in model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaModel.from_pretrained(model_name)
        elif 'deberta' in model_name.lower():
            self.tokenizer = DebertaTokenizer.from_pretrained(model_name)
            self.model = DebertaModel.from_pretrained(model_name)
        else:  # 默认使用BERT
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, text):
        """Encode text into BERT input format."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        return inputs

    def get_embeddings(self, inputs):
        """Get BERT embeddings for encoded inputs."""
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects.

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def write_vocab(vocab, filename):
    """Writes a vocab to a file.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    """
    print("Writing vocab...")
    with open(filename, "w", encoding="utf-8") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    f.close()
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file.

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename, encoding="utf-8") as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
        f.close()
    except IOError:
        raise MyIOError(filename)
    return d


def pad_sequences(sequences, pad_tok, max_length):
    """Pad sequences to a fixed length.

    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        max_length: maximum length of the sequence

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]
    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """Generate minibatches from data.

    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x]
        y_batch += [y]
    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunks(seq, tags):
    """
    给定一个标签序列，提取实体块及其位置。

    Args:
        seq: [4, 4, 0, 0, ...] 标签序列
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    """
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None

    for i, tok in enumerate(seq):
        # 忽略无效标签（-100）
        if tok == -100:
            continue

        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_chunk_type(tok, idx_to_tag):
    """Get chunk type from token and tag dictionary.

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


# bert = BERTProcessor()
# dict1 = load_vocab("D:\\PythonProject\\LiResolver_copy\\EE5\\data\\tags.txt")
# t = get_chunks([4, 4, 4, 0, 0], dict1)
# print(t)


def collate_fn(batch, tokenizer, tag2idx, max_length, device):
    """
    自定义 collate_fn，用于处理长度不一致的样本。
    """
    words, tags = zip(*batch)

    # 将每个句子（单词列表）转换为字符串
    words = [" ".join(word_seq) for word_seq in words]

    # 将 words 和 tags 转换为 BERT 输入格式
    encoding = tokenizer(
        words,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 手动对齐标签序列的长度
    labels = []
    for tag_seq in tags:
        aligned_labels = []

        for j in range(input_ids.size(1)):  # 遍历每个 token
            if j < len(tag_seq):
                aligned_labels.append(tag2idx.get(tag_seq[j], tag2idx["O"]))
            else:
                aligned_labels.append(-100)  # 填充部分使用 -100
        labels.append(aligned_labels)
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 修改后的collate_fn_new
def collate_fn_new(batch, tokenizer, tag2idx, max_length, device):
    words_batch = [item[0] for item in batch]
    tags_batch = [item[1] for item in batch]
    input_ids, attention_masks, labels = [], [], []

    for words, tags in zip(words_batch, tags_batch):
        # 动态计算分词并处理 max_length 截断问题
        tokens = []
        word_subtoken_counts = []
        current_len = 0
        for word in words:
            subtokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            if current_len + len(subtokens) > max_length - 2:
                break
            tokens.extend(subtokens)
            word_subtoken_counts.append(len(subtokens))
            current_len += len(subtokens)
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

        # 转换 input_ids 和 attention_mask
        input_ids_seq = tokenizer.convert_tokens_to_ids(tokens)
        input_ids_padded = input_ids_seq + [tokenizer.pad_token_id] * (max_length - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))

        # 处理标签
        aligned_labels = [-100] * max_length
        current_pos = 1  # CLS 位置为0
        tag_idx = 0
        for count in word_subtoken_counts:
            if tag_idx >= len(tags) or current_pos + count > max_length - 1:
                break
            original_tag = tags[tag_idx]
            # 首子词用原标签
            aligned_labels[current_pos] = tag2idx.get(original_tag, tag2idx["O"])
            # 后续子词用 I-标签，若不存在则回退到原标签
            for i in range(1, count):
                if current_pos + i >= max_length - 1:
                    break
                i_tag = original_tag if "-" not in original_tag else f"I-{original_tag.split('-', 1)[1]}"
                aligned_labels[current_pos + i] = tag2idx.get(i_tag, tag2idx["O"])
            current_pos += count
            tag_idx += 1

        # 保证长度正确
        input_ids.append(torch.tensor(input_ids_padded, dtype=torch.long))
        attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
        labels.append(torch.tensor(aligned_labels, dtype=torch.long))

    return {
        "input_ids": torch.stack(input_ids).to(device),
        "attention_mask": torch.stack(attention_masks).to(device),
        "labels": torch.stack(labels).to(device)
    }


def collate_fn_roberta(batch, tokenizer, tag2idx, max_length, device):
    words_batch = [item[0] for item in batch]
    tags_batch = [item[1] for item in batch]
    input_ids, attention_masks, labels = [], [], []

    for words, tags in zip(words_batch, tags_batch):
        # RoBERTa 使用不同的特殊标记
        tokens = []
        word_subtoken_counts = []
        current_len = 0
        for word in words:
            # RoBERTa tokenizer 可能返回空列表，需要处理这种情况
            subtokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
            if current_len + len(subtokens) > max_length - 2:  # RoBERTa 也需要 2 个特殊标记
                break
            tokens.extend(subtokens)
            word_subtoken_counts.append(len(subtokens))
            current_len += len(subtokens)
        
        # 使用 RoBERTa 的特殊标记
        tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]

        # 其余处理逻辑保持不变
        input_ids_seq = tokenizer.convert_tokens_to_ids(tokens)
        input_ids_padded = input_ids_seq + [tokenizer.pad_token_id] * (max_length - len(tokens))
        attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))

        # 标签对齐逻辑保持不变
        aligned_labels = [-100] * max_length
        current_pos = 1  # 第一个标记位置
        tag_idx = 0
        for count in word_subtoken_counts:
            if tag_idx >= len(tags) or current_pos + count > max_length - 1:
                break
            original_tag = tags[tag_idx]
            aligned_labels[current_pos] = tag2idx.get(original_tag, tag2idx["O"])
            for i in range(1, count):
                if current_pos + i >= max_length - 1:
                    break
                i_tag = original_tag if "-" not in original_tag else f"I-{original_tag.split('-', 1)[1]}"
                aligned_labels[current_pos + i] = tag2idx.get(i_tag, tag2idx["O"])
            current_pos += count
            tag_idx += 1

        input_ids.append(torch.tensor(input_ids_padded, dtype=torch.long))
        attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
        labels.append(torch.tensor(aligned_labels, dtype=torch.long))

    return {
        "input_ids": torch.stack(input_ids).to(device),
        "attention_mask": torch.stack(attention_masks).to(device),
        "labels": torch.stack(labels).to(device)
    }


