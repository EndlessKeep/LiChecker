import os
from .general_utils import get_logger
from .data_utils import load_vocab


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load: (bool) if True, load vocabulary and other configurations
        """
        # Directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # Create instance of logger
        self.logger = get_logger(self.path_log)
        self.tag2idx = {}
        # Load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary and other configurations"""
        # 1. Vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.vocab_tags)}
        self.nwords = len(self.vocab_words)
        self.ntags = len(self.vocab_tags)

        # 2. BERT configuration
        self.bert_model_name = r'D:\PythonProject\LiResolver_copy\EE5\data\legal-bert-ner'  # BERT model name
        self.max_length = 512  # Maximum sequence length for BERT

    # General config
    file_path = "D:\\PythonProject\\LiResolver_copy\\EE5\\"
    dir_output = file_path + "results\\test\\"
    dir_model = dir_output + "model_6_19\\"
    path_log = dir_output + "log.txt"

    # Dataset paths
    filename_train = file_path + "data\\train.txt"
    filename_dev = file_path + "data\\dev.txt"
    filename_test = file_path + "data\\test.txt"
    filename_dir_test = file_path + "data\\test\\"
    filename_dir_pre = file_path + "data\\test-pre\\"

    # Vocab files (created from dataset with build_data.py)
    filename_words = file_path + "data\\words.txt"
    filename_tags = file_path + "data\\tags.txt"

    # Training configuration
    train_embeddings = False  # Not needed for BERT
    nepochs = 10  # Number of epochs
    dropout = 0.15  # Dropout rate
    batch_size = 16  # Batch size
    lr_method = "adam"  # Learning rate method
    lr = 1e-5  # Learning rate
    lr_decay = 0.9  # Learning rate decay
    clip = -1  # Gradient clipping (if negative, no clipping)
    nepoch_no_imprv = 50  # Early stopping patience

    # Model hyperparameters
    hidden_size_lstm = 0  # LSTM hidden size (if used)
    hidden_size = 256
    use_crf = False  # Whether to use CRF
    # 注意力机制相关配置
    use_attention = True  # 是否使用注意力机制
    num_attention_heads = 4  # 注意力头数量
    attention_dropout = 0.1  # 注意力dropout率