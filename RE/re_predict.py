"""Experiment-running framework."""
import argparse
import importlib
import json
from logging import debug
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from pytorch_lightning.trainer import training_tricks
import torch
import torch.utils.data as Data
import pytorch_lightning as pl
#import lit_models
import yaml
import time
from transformers import BertTokenizerFast
from sklearn.metrics import confusion_matrix
#from lit_models import TransformerLitModelTwoSteps
from transformers import AutoConfig, AutoModel
#from pytorch_lightning.plugins import DDPPlugin
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ###


# from . import data
# from . import models
# from . import lit_models


# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(name=module_name, package='.')
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)  # False
    parser.add_argument("--litmodel_class", type=str, default="BertLitModel")  # TransformerLitModel
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="WIKI80")  # DIALOGUE
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="RobertaForPrompt")  # bert.BertForSequenceClassification
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--ossl2_label_type", type=str, default="relation")  # relation or tail

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()

    # data_class = _import_class(f"data.{temp_args.data_class}") ###
    # model_class = _import_class(f"models.{temp_args.model_class}")
    # litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")
    # data_class = _import_class(f"{temp_args.data_class}")  ###
    # model_class = _import_class(f"{temp_args.model_class}")
    # litmodel_class = _import_class(f"{temp_args.litmodel_class}")
    # import data
    from data.dialogue import WIKI80 as data_class
    from models import RobertaForPrompt as model_class
    from lit_models.transformer import BertLitModel as litmodel_class

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def load_re_model():
    parser = _setup_parser()
    args = parser.parse_args()

    print('args:', args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    # model_class = _import_class(f"models.{args.model_class}")
    # litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")
    # data_class = _import_class(f"data.{args.data_class}")
    from data.dialogue import WIKI80 as data_class
    from models import RobertaForPrompt as model_class
    from lit_models.transformer import BertLitModel as litmodel_class

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    data = data_class(args, model)
    data_config = data.get_data_config()

    model.resize_token_embeddings(len(data.tokenizer))

    data.setup_1()

    ### 搭建lit_model
    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
    data.tokenizer.save_pretrained('test')

    ### 读取已经训练好的model
    device = torch.device("cuda")

    print('torch.cuda.is_available(): ', torch.cuda.is_available())  # torch.cuda.is_available():  True

    best_model_path = 'D:\\PythonProject\\LiResolver_copy\\RE\\checkpoints\\f1=0.97.ckpt'
    # lit_model.load_state_dict(torch.load(best_model_path, map_location="cuda:0")["state_dict"])
    # lit_model.to(device) ###
    lit_model.load_state_dict(torch.load(best_model_path)["state_dict"])
    lit_model.to(device)
    print(next(lit_model.parameters()).device)

    return args, lit_model


def plot_confusion_matrix(truth_labels, preds, class_names):
    cm = confusion_matrix(truth_labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.show()


def plot_confusion_matrix_1(truth_labels, preds, class_names):
    # 过滤掉 other (0) 和 padding (1) 的标签和预测结果
    valid_indices = (truth_labels != 0) & (truth_labels != 1)
    filtered_truth = truth_labels[valid_indices]
    filtered_preds = preds[valid_indices]
    filtered_class_names = class_names  # 移除 "Other" 和 "padding"

    # 计算混淆矩阵
    cm = confusion_matrix(filtered_truth, filtered_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(18, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_class_names,
                yticklabels=filtered_class_names, annot_kws={"size": 18})
    plt.xticks(rotation=0, ha='center',fontsize=16)  # 将 x 轴标签旋转 45 度，并右对齐
    plt.yticks(rotation=0,fontsize=16)
    #plt.xlabel('Predicted', fontsize=24)
    #plt.ylabel('Truth', fontsize=24)
    #plt.title('Confusion Matrix',fontsize=28)
    plt.tight_layout(pad=2.0)
    plt.savefig('D:\\PythonProject\\LiResolver_copy\\RE\\dataset\\ossl2\\confusion_matrix_1.png',dpi=300)
    plt.show()

def predict_re(args, lit_model, relation_to_label):
    # model_class = _import_class(f"models.{args.model_class}")
    # data_class = _import_class(f"data.{args.data_class}")
    from data.dialogue import WIKI80 as data_class
    from models import RobertaForPrompt as model_class

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    data = data_class(args, model)
    data_config = data.get_data_config()

    # 读取测试数据 放进来
    data.setup_3()

    TPLG = []
    PRED = []
    TRUTH = []
    device = next(lit_model.parameters()).device
    ''' 预测 '''
    model.eval()
    with torch.no_grad():
        loader = data.val_dataloader()
        print(f"Total batches in loader: {len(loader)}")
        for batch in loader:
            # for ins in batch:
            #     print(ins)
            #     print(ins[0])
            #     print(ins[1])
            #     print(ins[2])
            A = batch['input_ids']
            B = batch['attention_mask']
            C = batch['labels']
            input_ids = torch.cat([torch.unsqueeze(ins, dim=0) for ins in A]).to(device)
            attention_mask = torch.cat([torch.unsqueeze(ins, dim=0) for ins in B]).to(device)
            labels = torch.cat([torch.tensor(np.expand_dims(ins, 0)) for ins in C]).to(device)
            # print('input_ids', input_ids.size())  # torch.Size([10, 256])
            # print('attention_mask', attention_mask.size())  # torch.Size([10, 256])
            # print('labels', labels.size())  # torch.Size([10])

            # 预测
            logits = lit_model.model(input_ids, attention_mask,
                                     return_dict=True).logits  #### torch.Size([10, 256, 50295])
            logits = lit_model.pvp(logits, input_ids)  # 在各个类别上的概率 # torch.Size([10, 19])
            test_pre_logits = logits.detach().cpu().numpy()  # (10, 19)
            preds = np.argmax(test_pre_logits, axis=-1)  # 预测的标签 # (10,) [ 4 12 13 13 13 13 14 13 13 13]
            # test_labels = labels.detach().cpu().numpy() # 实际的标签 # (10,) [7 7 7 7 7 7 7 7 7 7]
            # print('test_pre_logits: ', test_pre_logits.shape)
            # print('preds: ', preds.shape, preds)
            labels = labels.detach().cpu().numpy()
            TPLG.extend(test_pre_logits)
            PRED.extend(preds.tolist())
            TRUTH.extend(labels.tolist())
            # print(f"Batch processed. PRED length: {len(PRED)}, TPLG length: {len(TPLG)}")
    TPLG = np.concatenate(TPLG, axis=0)
    PRED = np.array(PRED)
    TRUTH = np.array(TRUTH)
    print(f"Final TRUTH length: {len(TRUTH)}")
    print(f"Final PRED length: {len(PRED)}")
    print(f"Final TPLG shape: {TPLG.shape}")
    with open('D:\\PythonProject\\LiResolver_copy\\RE\\dataset\\ossl2\\test.txt', 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    truth_labels = [relation_to_label[item['relation']] for item in test_data]
    min_length = min(len(truth_labels), len(PRED))
    truth_labels = truth_labels[:min_length]
    PRED = PRED[:min_length]
    plot_confusion_matrix_1(TRUTH, PRED, class_names)
    # return test_pre_logits, preds
    # print('TPLG: ', TPLG.shape)
    # print('PRED: ', PRED.shape)

    return TPLG, PRED
def predict_re_1(args, lit_model, file_entities):
    from data.dialogue import WIKI80 as data_class
    from models import RobertaForPrompt as model_class

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_model_name)
    relations = []
    for entity_pair in generate_entity_pairs(file_entities):
        input_text = f"{entity_pair['entity1']} [SEP] {entity_pair['entity2']}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        logits = lit_model(**inputs).logits
        logits = lit_model.pvp(logits, inputs["input_ids"])
        pred_label = np.argmax(logits.detach().cpu().numpy(), axis=-1)[0]
        relations.append({
            "entity1_type": entity_pair['entity1_type'],
            "entity2_type": entity_pair['entity2_type'],
            "relation": pred_label
        })
    return relations

def generate_entity_pairs(entities):
    pairs = []
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            pairs.append({
                "entity1": entities[i]["text"],
                "entity1_type": entities[i]["type"],
                "entity2": entities[j]["text"],
                "entity2_type": entities[j]["type"]
            })
    return pairs


# 假设你有一个类名列表
class_names = ["Action-Recipient(e1,e2)", "Action-Attitude(e1,e2)", "Action-Condition(e1,e2)",
               "Action-Exception(e1,e2)"]

relation_dict = {"Action-Recipient(e1,e2)": 0,
                 "Action-Attitude(e1,e2)": 1, "Action-Condition(e1,e2)": 2,
                 "Condition-Action(e1,e2)": 3}
re_args, re_model = load_re_model()
test_pre_logits, preds = predict_re(args=re_args, lit_model=re_model, relation_to_label=relation_dict)

print(test_pre_logits.shape)
print(preds.shape)
# import json
# with open('./re_test_preds.json', 'w', encoding="utf-8") as fw:
#     json.dump(preds, fw)
