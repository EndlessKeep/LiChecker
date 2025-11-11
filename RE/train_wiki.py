import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from transformers import AutoConfig, AutoTokenizer
import numpy as np
from data.dialogue import WIKI80
from models import RobertaForPrompt
from lit_models.transformer import BertLitModel

# 这个是训练py文件
def setup_parser():

    parser = argparse.ArgumentParser(add_help=False)

    # 训练器参数
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--litmodel_class", type=str, default="BertLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="WIKI80")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="RobertaForPrompt")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--ossl2_label_type", type=str, default="relation")


    temp_args, _ = parser.parse_known_args()

    from data.dialogue import WIKI80 as data_class
    from models import RobertaForPrompt as model_class
    from lit_models.transformer import BertLitModel as litmodel_class

    # data, model, LitModel
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def train():
    # 设置parser
    parser = setup_parser()
    args = parser.parse_args()

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    # 加载数据
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = RobertaForPrompt.from_pretrained(args.model_name_or_path, config=config)
    data = WIKI80(args, model)
    data.setup()
    model.resize_token_embeddings(len(data.tokenizer))
    tokenizer = data.get_tokenizer()
    model.update_word_idx(len(tokenizer))

    lit_model = BertLitModel(args=args, model=model, tokenizer=data.tokenizer)
    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5,
                                                check_on_train_epoch_end=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath="output",
        filename='{epoch}-{Eval/f1:.2f}',
        save_top_k=1,
        monitor="Eval/f1",
        mode="max",
        save_weights_only=True
    )
    callbacks = [early_callback, checkpoint_callback]

    logger = TensorBoardLogger("logs", name="roberta-wiki80")

    # Initialize the Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        gpus=1 if torch.cuda.is_available() else 0
    )

    # Train the model
    trainer.fit(lit_model, datamodule=data)

if __name__ == "__main__":
    train()