import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.plugins import DDPPlugin
from transformers import AutoConfig, AutoTokenizer
import numpy as np
from data.dialogue import WIKI80,DIALOGUE
from models import RobertaForPrompt
from lit_models.transformer import BertLitModel,DialogueLitModel

# 这个不是训练py文件
def setup_parser():

    parser = argparse.ArgumentParser(add_help=False)


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

    from data.dialogue import DIALOGUE as data_class
    from models import RobertaForPrompt as model_class
    from lit_models.transformer import DialogueLitModel as litmodel_class

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def train():

    parser = setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = RobertaForPrompt.from_pretrained(args.model_name_or_path, config=config)
    data = DIALOGUE(args)
    data.setup()
    model.resize_token_embeddings(len(data.tokenizer))

    lit_model = DialogueLitModel(args=args, model=model, tokenizer=data.tokenizer)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="roberta-wiki80-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="Eval/loss",
        mode="min",
        save_weights_only=True
    )

    early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5,
                                                check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
                                                    filename='{epoch}-{Eval/f1:.2f}',
                                                    dirpath="output",
                                                    save_weights_only=True
                                                    )
    callbacks = [early_callback, model_checkpoint]

    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, default_root_dir="training/logs",
                                            gpus=gpu_count, accelerator=accelerator,
                                            plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
                                            )

    trainer.fit(lit_model, datamodule=data)

if __name__ == "__main__":
    train()