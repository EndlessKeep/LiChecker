import torch

from ner_model.data_utils import collate_fn_new,collate_fn
from ner_model.model import NERModel
from ner_model.config import Config
config = Config()
model = NERModel(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample = [(["New", "York", "City"], ["I-Condition", "I-Attitude", "B-Recipient"])]
collated = collate_fn_new(sample, model.tokenizer, config.tag2idx, 512, device)
print("Input IDs:", collated["input_ids"].shape)
print("Labels:", collated["labels"])
