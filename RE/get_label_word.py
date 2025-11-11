from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json


model_name_or_path = "roberta-base"
dataset_name = "ossl2"
tokenizer_path = 'D:\\PythonProject\\LiResolver_copy\\RE\\dataset\\ossl2\\roberta-base'
rel2id_new_path = 'D:\\PythonProject\\LiResolver_copy\\RE\\dataset\\ossl2\\rel2id_new.json'
output_ossl = 'D:\\PythonProject\\LiResolver_copy\\RE\\dataset'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
def split_label_words(tokenizer, label_list):
    label_word_list = []
    for label in label_list:
        if label == 'no_relation' or label == "NA":
            label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
            label_word_list.append(torch.tensor(label_word_id))
        else:
            tmps = label
            label = label.lower()
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list

with open(rel2id_new_path, "r") as file:
    t = json.load(file)
    label_list = list(t)

t = split_label_words(tokenizer, label_list)

with open(f"{output_ossl}\\{model_name_or_path}_{dataset_name}_new.pt", "wb") as file:
    torch.save(t, file)
