# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.trainer_utils import EvalPrediction

from .args import ScriptArguments
from .data import load_and_preprocess
from .reward_model import *

__all__ = ['train_pipeline']

# torch.backends.cudnn.enable = True
# torch.backends.cudnn.benchmark = True
tqdm.pandas()

DATA_PATH = {
    "total": "/root/dataset/anthropic-rlhf/total/train.json",

    "helpful": "/root/dataset/anthropic-rlhf/helpful-base/train.json",
    "hf": "/root/dataset/anthropic-rlhf/helpful-base/train.json",

    "harmless": "/root/dataset/anthropic-rlhf/harmless-base/train.json",
    "hl": "/root/dataset/anthropic-rlhf/harmless-base/train.json"
}


def get_suffix(p):
    idx = 1
    while Path(p + f"_Exp{idx}").exists():
        idx += 1
    return f"_Exp{idx}"


def get_args():
    args = tyro.cli(ScriptArguments)
    args.reward_config.local_rank = args.local_rank

    output_name = args.model_name.rsplit("/", 1)[-1]+'_'+args.dataset_name
    if not args.test:
        args.output_dir = args.output_dir.format(output_name)
        args.reward_config.logging_dir = args.reward_config.logging_dir.format(output_name)
        args.reward_config.output_dir = args.reward_config.output_dir.format(output_name)
        suffix = get_suffix(args.output_dir)
        args.output_dir += suffix
        args.reward_config.logging_dir += suffix
        args.reward_config.output_dir += suffix
        print("logging dir:", args.reward_config.logging_dir)
    else:
        args.output_dir = args.output_dir.format("TEST")
        args.reward_config.logging_dir = args.reward_config.logging_dir.format("TEST")
        args.reward_config.output_dir = args.reward_config.output_dir.format("TEST")
    args.reward_config.per_device_eval_batch_size = args.reward_config.per_device_train_batch_size
    return args


def load_model_and_tokenizer(args):
    if not args.test:
        import json
        import dataclasses
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir)/"config.txt", "w") as f:
            f.write(json.dumps(dataclasses.asdict(args), indent=4))
    if args.MoRM:
        model = MoRMForPreGating(
            args.model_name,
            num_expert=args.morm_config.num_expert,
            trust_remote_code=True
        )
    else:
        if args.dispatch.cuda_list is not None:
            model = RewardModel(
                args.model_name, 
                dispatch=True, 
                cuda_list=args.dispatch.cuda_list, 
                memory=args.dispatch.memory, 
                trust_remote_code=True,
                bias=args.bias
            )
        else:
            model = RewardModel(
                args.model_name,
                trust_remote_code=args.trust_remote_code,
                bias=args.bias
            )
    
    print(model.state_dict)

    # Load the dataset and pre-process it
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if "pythia" in args.model_name or "phi-2" in args.model_name:
        tokenizer.padding_side="left"
        tokenizer.add_special_tokens({'pad_token': ' '})
    return model, tokenizer


def get_trainer_and_train(args, model, tokenizer, train_dataset, eval_dataset):
    def collate_fn(batch):
        ITEM = {
            k: torch.tensor([item[k] for item in batch]).squeeze(1) for k in ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"]
        }
        if "consensus" in batch[0]:
            ITEM["consensus"] = torch.tensor([item["consensus"] for item in batch], dtype=torch.int)
        return ITEM
    
    def compute_metrics(pre: EvalPrediction):
        """
        results will be denumpified and detensorized later in the evaluation loop, so it will be fine not to remove these two types.
        """
        p = pre.predictions
        label = pre.label_ids
        acc = np.mean(p > 0)
        reward_diff = np.mean(p)
        if (label == 1).all():
            return {"acc": 100. * acc, "difference": reward_diff}
        agreed_idx = (label == 1)
        disagreed_idx = (label != 1)
        ap = p[agreed_idx]
        dp = p[disagreed_idx]
        a_acc = np.mean(ap > 0)
        a_rd = np.mean(ap)
        a_l = np.size(ap)
        d_acc = np.mean(dp > 0)
        d_rd = np.mean(dp)
        d_l = np.size(dp)

        return {"acc": 100. * acc, "difference": reward_diff,
                "agreed_acc": 100. * a_acc, "agreed_difference": a_rd, "agreed_num": a_l,
                "disagreed_acc": 100. * d_acc, "disagreed_difference": d_rd, "disagreed_num": d_l}

    if args.dispatch.cuda_list is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.dispatch.cuda_list[0]
        print("CVD:", os.environ["CUDA_VISIBLE_DEVICES"])
    if args.MoRM:
        if args.use_em:
            trainer = PairwiseTrainerForMoRMwithEM(
                model=model,
                tokenizer=tokenizer,
                args=args.reward_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collate_fn,
                save_steps=args.save_steps,
                output_dir=args.output_dir,
                compute_metrics=compute_metrics,
                morm_config=args.morm_config
            )
        else:
            trainer = PairwiseTrainerForMoRMwithSoftmax(
                model=model,
                tokenizer=tokenizer,
                args=args.reward_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collate_fn,
                save_steps=args.save_steps,
                output_dir=args.output_dir,
                compute_metrics=compute_metrics,
                morm_config=args.morm_config
            )
    else:
        trainer = PairwiseTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args.reward_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            compute_metrics=compute_metrics,
        )
    output = trainer.train()
    if not args.not_save_model:
        torch.save(model.state_dict(), args.output_dir+"/final.pth")


def train_pipeline(args=None):
    if args is None:
        args = get_args()
    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset, eval_dataset = load_and_preprocess(args, tokenizer, split_from_train_ratio=0.1)
    get_trainer_and_train(args, model, tokenizer, train_dataset, eval_dataset)

