# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import inspect
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
# from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from trl import is_peft_available
from trl import RewardConfig
from trl.trainer.utils import PeftSavingCallback, RewardDataCollatorWithPadding, compute_accuracy

from .args import MoRMConfig


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


NO_SPLIT_MODULE_MAPPING = {
    "chatglm": ["GLMBlock"],
    "phi-2": ["CodeGenBlock"],
}


def load_model(mpath, cuda_list="0,1,2", memory="30GiB", host_low_loading=False):
    cuda_list = cuda_list.split(',')
    no_split_module_classes = []
    for k, v in NO_SPLIT_MODULE_MAPPING.items():
        if k in mpath:
            no_split_module_classes.extend(v)
    max_memory = {int(cuda): memory for cuda in cuda_list}
    if host_low_loading:
        max_memory[int(cuda_list[0])] = "1GiB"
    config = AutoConfig.from_pretrained(mpath, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes)
    load_checkpoint_in_model(model, mpath, device_map=device_map)
    model = dispatch_model(model,device_map=device_map)
    return model


def fetch(key, dic, default):
    if key in dic:
        return dic.pop(key)
    else:
        return default


class RewardModel(nn.Module):
    def __init__(self, mpath, *args, dim_head=1, dispatch=False, cuda_list=None, memory=None, bias=True, **kwargs) -> None:
        super().__init__()
        if dispatch:
            self.model = load_model(mpath, cuda_list=cuda_list, memory=memory, host_low_loading=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(mpath, *args, torch_dtype=torch.float16, **kwargs)
        tmp = next(self.model.parameters())
        self.device = tmp.device
        self.value_head = nn.Linear(self.model.config.hidden_size, dim_head, dtype=tmp.dtype, bias=bias).to(tmp.device)    
        self.one_score = True if dim_head==1 else False

    def forward(self, sequences: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len]
        sequences = sequences.to("cuda")
        attention_mask = attention_mask.to("cuda")
        outputs = self.model(sequences, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        last_hidden_states = outputs.hidden_states[-1]
        sequence_lengths = torch.max(attention_mask * torch.arange(sequences.size(1), device=attention_mask.device).unsqueeze(0), dim=1)[0]
        sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), sequence_lengths]
        if self.one_score:
            return self.value_head(sequence_hidden_states).squeeze(1)  # ensure shape is (B, )
        return self.value_head(sequence_hidden_states.to(self.value_head.bias.device))
        

class ProbalisticRewardExpert(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        dtype = torch.float16 if "quantization_config" in kwargs and kwargs["quantization_config"] is not None else None
        self.model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
        self.mu_head = nn.Linear(self.model.config.hidden_size, 1, dtype=dtype)
        self.logvar_head = nn.Linear(self.model.config.hidden_size, 1, dtype=dtype)
        self.device = None

    def forward(self, sequences: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len]
        # assert sequences.shape == (2, 512) and attention_mask.shape == (2, 512), f"{sequences.shape} {attention_mask.shape}"
        
        outputs = self.model(sequences, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        last_hidden_states = outputs.hidden_states[-1]
        sequence_lengths = torch.max(attention_mask * torch.arange(sequences.size(1), device=sequences.device).unsqueeze(0), dim=1)[0]
        sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), sequence_lengths]
        mu = self.mu_head(sequence_hidden_states)
        logvar = self.logvar_head(sequence_hidden_states)
        return mu, logvar


class MoPRE(nn.Module):
    def __init__(self, *args, num_expert=3, sampling_strategy="no", **kwargs):
        super().__init__()
        self.num_expert = num_expert
        self.sampling_strategy = sampling_strategy
        for i in range(self.num_expert):
            setattr(self, f"expert{i+1}", ProbalisticRewardExpert(*args, **kwargs))

    def forward(self, sequences, attention_mask, no_mle=False):
        mu, logvar = list(zip(*[getattr(self, f"expert{i+1}")(sequences, attention_mask) for i in range(self.num_expert)]))
        # [B, K]
        mu = torch.concat(mu, dim=1)
        logvar = torch.concat(logvar, dim=1)
        if no_mle:
            return self.sample(mu, logvar)
        mu, logvar = self.sample(mu, logvar)
        divvar = torch.exp(-logvar)
        return torch.sum(mu * divvar, dim=1, keepdim=True) / torch.sum(divvar, dim=1, keepdim=True)

    def sample(self, mu, logvar, sampling_strategy=None, **kwargs):
        ss = sampling_strategy if sampling_strategy is not None else self.sampling_strategy
        if ss == "no":
            return mu, logvar
        elif ss == "most_k_probable":
            raise NotImplementedError
        else:
            raise NotImplementedError
        

class MoRMForPreGating(nn.Module):
    def __init__(self, *args, num_expert=2, **kwargs) -> None:
        super().__init__()
        for i in range(num_expert):
            setattr(self, f"expert{i+1}", RewardModel(*args, **kwargs))
        self.num_expert = num_expert
        self.gating = RewardModel(*args, dim_head=num_expert, **kwargs)
        
    def forward(self, sequences: torch.LongTensor, attention_mask: torch.Tensor, return_all=False) -> torch.Tensor:
        u = torch.concat([getattr(self, f'expert{i+1}')(sequences, attention_mask).unsqueeze(1) for i in range(self.num_expert)], dim=1)
        if return_all:
            likelihood = self.gating(sequences, attention_mask)
            return u, likelihood
        idx = torch.argmax(self.gating(sequences, attention_mask), dim=1)
        return u[torch.arange(u.size(0)), idx]


class MoERMForPreGating(nn.Module):
    def __init__(self, *args, num_expert=5, **kwargs) -> None:
        super().__init__()
        for i in range(num_expert):
            setattr(self, f"expert{i+1}", RewardModel(*args, **kwargs))
        self.num_expert = num_expert
        self.gating = RewardModel(*args, dim_head=num_expert, **kwargs)
        self.eps = 1e-5

    def load_expert_via_multi_file(self, path, ids):
        for i, model_path in enumerate(ids):
            getattr(self, f"expert{i+1}").load_state_dict(torch.load(path.format(model_path)))

    def forward(self, sequences: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        u = torch.concat([getattr(self, f'expert{i+1}')(sequences, attention_mask).unsqueeze(1) for i in range(self.num_expert)], dim=1)
        std = self.gating(sequences, attention_mask)
        divvar = 1 / (std**2 + self.eps)
        # MLE
        pre = torch.sum(u * divvar, dim=1) / torch.sum(divvar, dim=1)
        return pre


class MoERMForIntermediateGating(nn.Module):
    def __init__(self):
        pass
    

class RewardTrainerForMLERegression(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.loss_fn = nn.MSELoss(reduction="mean")

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        predict = model(inputs["input_ids"], inputs["attention_mask"])
        loss = self.loss_fn(predict, inputs["score"])

        if return_outputs:
            return loss, predict
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        return loss, None, predict


class RewardTrainerForPretrain(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.loss_fn = nn.MSELoss(reduction="mean")

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        predict = model(inputs["input_ids"], inputs["attention_mask"])
        loss = self.loss_fn(predict, inputs["score"])

        if return_outputs:
            return loss, predict
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        return loss, None, predict


class PairwiseTrainerForMoRMwithSoftmax(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        save_steps: int = 99999,
        output_dir: str = "",
        morm_config: MoRMConfig = None
    ):
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.count = 0
        assert save_steps > 1
        self.save_steps = save_steps
        self.model_dir = output_dir
        self.div_eps = 1e-3
        self.logging_data = defaultdict(list)
        self.log_count = args.logging_steps * args.gradient_accumulation_steps
        if morm_config.method == "softmax":
            self.pretrain_threshold = int(morm_config.pretrain_ratio * len(train_dataset) / args.per_device_train_batch_size)
            print(self.pretrain_threshold)
            self.compute_loss = self._compute_loss_softmax
        elif morm_config.method == "gumbel-softmax":
            self.tau = morm_config.tau
            self.gamma = morm_config.gamma
            self.compute_loss = self._gumbel_wrapper(self._compute_loss_gumbel, hard=False)
        elif morm_config.method == "gumbel-argmax":
            self.tau = morm_config.tau
            self.gamma = morm_config.gamma
            self.compute_loss = self._gumbel_wrapper(self._compute_loss_gumbel, hard=True)
        elif morm_config.method == "argmax":
            self.pretrain_threshold = int(morm_config.pretrain_ratio * len(train_dataset) / args.per_device_train_batch_size)
            print(self.pretrain_threshold)
            self.compute_loss = self._compute_loss_argmax
            self.cls_loss_func = nn.CrossEntropyLoss()
            self.cls_ratio = morm_config.cls_loss_ratio


    def _compute_loss_softmax(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not return_outputs:
            self.count += 1
            if self.count % self.save_steps == 0:
                torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")
        
        mu_w, p_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], return_all=True)
        mu_l, p_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], return_all=True)
        dif = mu_w - mu_l
        
        if self.count < self.pretrain_threshold: 
            # pretrain
            dif_ = dif
            reward_loss = -nn.functional.logsigmoid(dif_).mean()
        else:
            dif_ = (mu_w*nn.functional.softmax(p_w, dim=1) - mu_l*nn.functional.softmax(p_l, dim=1)).sum(1)
            reward_loss = -nn.functional.logsigmoid(dif_).mean()

        # 0.69315 : E(RM(w) - RM(l)) = 0
        if return_outputs:
            return reward_loss, {"rewards_chosen": (mu_w*nn.functional.softmax(p_w, dim=1)).sum(1),
                "rewards_rejected": (mu_l*nn.functional.softmax(p_l, dim=1)).sum(1),
            }
        self.log({
            "Expert1": dif[:, 0].mean().cpu().item(), 
            "Expert2": dif[:, 1].mean().cpu().item(), 
            "pre_dif": dif_.mean().cpu().item(),
            "state": 0 if self.count < self.pretrain_threshold else 1
        })
        return reward_loss

    def _gumbel_wrapper(self, f, hard=True):
        def func(*args, **kwargs):
            return f(*args, hard=hard, **kwargs)
        return func

    def _compute_loss_gumbel(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        hard=True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not return_outputs:
            self.count += 1
            if self.count % self.save_steps == 0:
                torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")
        self.tau = max(self.tau * self.gamma, 0.1)
        mu_w, p_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], return_all=True)
        mu_l, p_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], return_all=True)
        dif = mu_w - mu_l
        p_w = F.gumbel_softmax(p_w, tau=self.tau, hard=hard)
        p_l = F.gumbel_softmax(p_l, tau=self.tau, hard=hard)
        dif_ = (mu_w*p_w - mu_l*p_l).sum(1)
        reward_loss = -nn.functional.logsigmoid(dif_).mean()

        # 0.69315 : E(RM(w) - RM(l)) = 0
        if return_outputs:
            return reward_loss, {"rewards_chosen": (mu_w*nn.functional.softmax(p_w, dim=1)).sum(1),
                "rewards_rejected": (mu_l*nn.functional.softmax(p_l, dim=1)).sum(1),
            }
        with torch.no_grad():
            max_dif = mu_w.max(1)[0] - mu_l.min(1)[0]
            min_dif = mu_w.min(1)[0] - mu_l.max(1)[0]
            precision = (dif_ - min_dif) / (max_dif - min_dif)
        self.record({
            "Expert1": dif[:, 0].mean().cpu().item(), 
            "Expert2": dif[:, 1].mean().cpu().item(), 
            "pre_dif": dif_.mean().cpu().item(),
            "precision": precision.mean().cpu().item(),
            "tau": self.tau
        }, not(self.count % self.log_count))
        return reward_loss

    def record(self, data_dict, log=False):
        for k, v in data_dict.items():
            self.logging_data[k].append(v)
        if log:
            self.log({k: sum(v)/len(v) for k, v in self.logging_data.items()})
            self.logging_data.clear()

    def _compute_loss_argmax(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not return_outputs:
            self.count += 1
            if self.count % self.save_steps == 0:
                torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")
        mu_w, p_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], return_all=True)
        mu_l, p_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], return_all=True)
        dif = mu_w - mu_l
        bs = dif.size(0)
        counting_arr = torch.arange(bs, dtype=torch.int).to(dif.device)
        
        if self.count < self.pretrain_threshold: 
            # pretrain
            dif_ = dif
            reward_loss = -nn.functional.logsigmoid(dif_).mean()
            cls_loss = 0
        else:
            gt_w = torch.argmax(mu_w, dim=-1).detach()
            gt_l = torch.argmin(mu_l, dim=-1).detach()
            cls_loss = self.cls_loss_func(p_w, gt_w) + self.cls_loss_func(p_l, gt_l)
            gate_w = torch.argmax(p_w, dim=-1).detach()
            gate_l = torch.argmax(p_l, dim=-1).detach()
            dif_ = mu_w[counting_arr, gate_w] - mu_l[counting_arr, gate_l]
            reward_loss = -nn.functional.logsigmoid(dif_).mean()

        # 0.69315 : E(RM(w) - RM(l)) = 0
        if return_outputs:
            return reward_loss, {"rewards_chosen": (mu_w*nn.functional.softmax(p_w, dim=1)).sum(1),
                "rewards_rejected": (mu_l*nn.functional.softmax(p_l, dim=1)).sum(1),
            }
        self.log({
            "Expert1": dif[:, 0].mean().cpu().item(), 
            "Expert2": dif[:, 1].mean().cpu().item(), 
            "pre_dif": dif_.mean().cpu().item(),
            "state": 0 if self.count < self.pretrain_threshold else 1
        })
        return reward_loss + self.cls_ratio * cls_loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)
        predict = predict["rewards_chosen"] - predict["rewards_rejected"]
        return loss, predict, torch.ones_like(predict)


class PairwiseTrainerForMoRMwithEM(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        save_steps: int = 99999,
        output_dir: str = "",
        morm_config: MoRMConfig = None
    ):
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.count = 0
        assert save_steps > 1
        self.save_steps = save_steps
        self.model_dir = output_dir
        self.div_eps = 1e-3
        if morm_config.em_ratio:
            self.em_step = int(morm_config.em_ratio * len(train_dataset) * args.num_train_epochs / args.per_device_train_batch_size)
        elif morm_config.em_step:
            self.em_step = int(morm_config.em_step)
        else:
            raise ValueError("em-step and em-ratio are both null.")
        print(self.em_step)
        if morm_config.method == "softmax":
            self.compute_loss = self._compute_loss_softmax
        elif morm_config.method == "argmax":
            raise NotImplementedError
            self.compute_loss = self._compute_loss_argmax
            self.cls_loss_func = nn.CrossEntropyLoss()
            self.cls_ratio = morm_config.cls_loss_ratio


    def _compute_loss_softmax(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not return_outputs:
            self.count += 1
            if self.count % self.save_steps == 0:
                torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")
        
        mu_w, p_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], return_all=True)
        mu_l, p_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], return_all=True)
        dif = mu_w - mu_l
        
        if self.count // self.em_step % 2: 
            # M step
            dif_ = (mu_w*nn.functional.softmax(p_w.detach(), dim=1) - mu_l*nn.functional.softmax(p_l.detach(), dim=1)).sum(1)
            reward_loss = -nn.functional.logsigmoid(dif_).mean()
        else:
            # E step
            dif_ = (mu_w.detach()*nn.functional.softmax(p_w, dim=1) - mu_l.detach()*nn.functional.softmax(p_l, dim=1)).sum(1)
            reward_loss = -nn.functional.logsigmoid(dif_).mean()

        # 0.69315 : E(RM(w) - RM(l)) = 0
        if return_outputs:
            return reward_loss, {"rewards_chosen": (mu_w*nn.functional.softmax(p_w, dim=1)).sum(1),
                "rewards_rejected": (mu_l*nn.functional.softmax(p_l, dim=1)).sum(1),
            }
        self.log({
            "Expert1": dif[:, 0].mean().cpu().item(), 
            "Expert2": dif[:, 1].mean().cpu().item(), 
            "pre_dif": dif_.mean().cpu().item(),
            "state": self.count // self.em_step % 2
        })
        return reward_loss

    def _compute_loss_argmax(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not return_outputs:
            self.count += 1
            if self.count % self.save_steps == 0:
                torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")
        mu_w, p_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], return_all=True)
        mu_l, p_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], return_all=True)
        dif = mu_w - mu_l
        bs = dif.size(0)
        counting_arr = torch.arange(bs, dtype=torch.int).to(dif.device)
        
        if self.count < self.pretrain_threshold: 
            # pretrain
            dif_ = dif
            reward_loss = -nn.functional.logsigmoid(dif_).mean()
            cls_loss = 0
        else:
            gt_w = torch.argmax(mu_w, dim=-1).detach()
            gt_l = torch.argmin(mu_l, dim=-1).detach()
            cls_loss = self.cls_loss_func(p_w, gt_w) + self.cls_loss_func(p_l, gt_l)
            gate_w = torch.argmax(p_w, dim=-1).detach()
            gate_l = torch.argmax(p_l, dim=-1).detach()
            dif_ = mu_w[counting_arr, gate_w] - mu_l[counting_arr, gate_l]
            reward_loss = -nn.functional.logsigmoid(dif_).mean()

        # 0.69315 : E(RM(w) - RM(l)) = 0
        if return_outputs:
            return reward_loss, {"rewards_chosen": (mu_w*nn.functional.softmax(p_w, dim=1)).sum(1),
                "rewards_rejected": (mu_l*nn.functional.softmax(p_l, dim=1)).sum(1),
            }
        self.log({
            "Expert1": dif[:, 0].mean().cpu().item(), 
            "Expert2": dif[:, 1].mean().cpu().item(), 
            "pre_dif": dif_.mean().cpu().item(),
            "state": 0 if self.count < self.pretrain_threshold else 1
        })
        return reward_loss + self.cls_ratio * cls_loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)
        predict = predict["rewards_chosen"] - predict["rewards_rejected"]
        return loss, predict, torch.ones_like(predict)


class RewardTrainerForPRM(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        alpha: float = 0.1,
        target_std: float = 0.5,
        save_steps: int = 99999,
        output_dir: str = ""
    ):
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.alpha = alpha
        self.target_std = target_std
        self.count = 0
        assert save_steps > 1
        self.save_steps = save_steps
        self.model_dir = output_dir
        self.div_eps = 1e-3

    def KL(self, mu, std, target_mu):
        """
        calculate D_{KL}(gt||predict)
        gt ~ N(target_mu, self.target_std^2)
        predict ~ N(mu, std^2)
        """
        # clipping !!
        var_ratio = (self.target_std / (std + self.div_eps)).pow(2)
        t1 = ((target_mu - mu) / (std + self.div_eps)).pow(2)
        return 0.5 * (var_ratio - var_ratio.log() - 1 + t1).mean()

    def KL_with_divvar(self, mu, divvar, target_mu):
        var_ratio = (self.target_std**2) * divvar
        t1 = (target_mu - mu).pow(2) * divvar
        return 0.5 * (var_ratio - (var_ratio + self.div_eps).log() - 1 + t1).mean()

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self.count += 1
        if self.count % self.save_steps == 0:
            torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")
        # CUDA memory limited
        # sampling are done
        # mu_w, sigma_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], no_mle=True)
        # mu_l, sigma_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], no_mle=True)
        # # [B, K]
        # w_1divvar = 1 / (sigma_w ** 2 + self.div_eps)
        # # [B, 1]
        # w_predict = torch.sum(mu_w * w_1divvar, dim=1, keepdim=True) / torch.sum(w_1divvar, dim=1, keepdim=True)
        # KL_w = self.KL(mu_w, sigma_w, w_predict)
        # l_1divvar = 1 / (sigma_l ** 2 + self.div_eps)
        # l_predict = torch.sum(mu_l * l_1divvar, dim=1, keepdim=True) / torch.sum(l_1divvar, dim=1, keepdim=True)
        # KL_l = self.KL(mu_l, sigma_l, l_predict)

        mu_w, logvar_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], no_mle=True)
        mu_l, logvar_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], no_mle=True)
        
        w_1divvar = torch.exp(-logvar_w)
        w_predict = torch.sum(mu_w * w_1divvar, dim=1, keepdim=True) / torch.sum(w_1divvar, dim=1, keepdim=True)
        KL_w = self.KL_with_divvar(mu_w, w_1divvar, w_predict)
        l_1divvar = torch.exp(-logvar_l)
        l_predict = torch.sum(mu_l * l_1divvar, dim=1, keepdim=True) / torch.sum(l_1divvar, dim=1, keepdim=True)
        KL_l = self.KL_with_divvar(mu_l, l_1divvar, l_predict)

        # 0.69315 : E(RM(w) - RM(l)) = 0
        reward_loss = -nn.functional.logsigmoid(w_predict - l_predict).mean()
        loss = reward_loss + self.alpha * (KL_w + KL_l)
        print(f"KL_w: {KL_w.item():.3f} KL_l: {KL_l.item():.3f} reward: {reward_loss.item():.3f} total: {loss.item(): .3f}")
        if return_outputs:
            return loss, {"rewards_chosen": w_predict,
                "rewards_rejected": l_predict,
            }
        # pdb.set_trace()
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        return loss, None, predict


class RewardTrainerForPRM_V2(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        alpha: float = 0.5,
        target_std: float = 0.5,
        max_target_std: Optional[float] = None,
        std_factor: float = 5,
        num_expert: int = 4,
        save_steps: int = 99999,
        output_dir: str = ""
    ):
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.alpha = alpha

        assert max_target_std is None or max_target_std > target_std, f"max_target_std({max_target_std}) should be greater than target_std({target_std})"
        assert max_target_std is not None or std_factor > 1, f"std_factor({std_factor}) should be greater than 1"
        if max_target_std is not None:
            print("When max_target_std and std_factor are set simultaneously, std_factor will be neglected")
        else:
            max_target_std = target_std * std_factor

        from math import log
        self.target_std_distribution = torch.linspace(log(target_std), log(max_target_std), num_expert).exp().unsqueeze(0).to("cuda")

        self.count = 0
        assert save_steps > 1
        self.save_steps = save_steps
        self.model_dir = output_dir
        self.div_eps = 1e-3
        self.idx_mat = torch.zeros((1, num_expert)).type(torch.int32)

    def sorted_KL_with_divvar(self, mu, divvar, target_mu, verbose_mode=False):
        """
        mu: [B, N]
        divvar: [B, N]
        target_mu: [B, N]
        calculate D_{KL}(label||predict)
        """
        # clipping !!
        var_ratio = self.target_std_distribution.pow(2) * divvar
        t1 = (target_mu - mu).pow(2) * divvar
        item_KL = 0.5 * (var_ratio - (var_ratio + self.div_eps).log() - 1 + t1)
        best_KL = item_KL[:, 0].mean()
        if verbose_mode:
            return best_KL + self.alpha * item_KL[:, 1:].sum(1).mean(), best_KL
        return best_KL + self.alpha * item_KL[:, 1:].sum(1).mean()

    def calculate_target_for_best_prediction(self, mu_w, logvar_w, mu_l, logvar_l):
        dif = mu_w - mu_l
        target_w = (mu_w + logvar_w.exp()/(dif.exp()+1)).detach()
        target_l = (mu_l - logvar_l.exp()/(dif.exp()+1)).detach()
        return target_w, target_l

    def sort_and_split(self, mu, logvar, descending=False):
        bs = mu.size(0)
        if self.idx_mat.size(0) < bs:
            self.idx_mat = torch.zeros_like(mu) + torch.arange(bs).unsqueeze(1).to(mu.device)
            self.idx_mat = self.idx_mat.type(torch.int32)
        mu_sort, idx_sort = torch.sort(mu, descending=descending)
        logvar_sort = logvar[self.idx_mat[:bs], idx_sort]
        return mu_sort, logvar_sort, mu_sort[:, :-1].detach()

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self.count += 1
        if self.count % self.save_steps == 0:
            torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")

        mu_w, logvar_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], no_mle=True)
        mu_l, logvar_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], no_mle=True)

        mu_w_sort, logvar_w_sort, mu_w_target = self.sort_and_split(mu_w, logvar_w, descending=True)
        mu_l_sort, logvar_l_sort, mu_l_target = self.sort_and_split(mu_l, logvar_l, descending=False)
        mu_w_best_target, mu_l_best_target = self.calculate_target_for_best_prediction(mu_w_sort[:, [0]], logvar_w_sort[:, [0]], mu_l_sort[:, [0]], logvar_l_sort[:, [0]])
        
        mu_w_target = torch.concat([mu_w_best_target, mu_w_target], dim=-1)
        mu_l_target = torch.concat([mu_l_best_target, mu_l_target], dim=-1)
        w_1divvar_sort = torch.exp(-logvar_w_sort)
        KL_w, reward_loss_w = self.sorted_KL_with_divvar(mu_w_sort, w_1divvar_sort, mu_w_target, verbose_mode=True)
        l_1divvar_sort = torch.exp(-logvar_l)
        KL_l, reward_loss_l = self.sorted_KL_with_divvar(mu_l_sort, l_1divvar_sort, mu_l_target, verbose_mode=True)

        # 0.69315 : RM(w) - RM(l) = 0
        reward_loss = reward_loss_w + reward_loss_l
        loss = KL_w + KL_l
        print(f"KL_w: {KL_w.item():.3f} KL_l: {KL_l.item():.3f} reward: {reward_loss.item():.3f} total: {loss.item(): .3f}")
        if return_outputs:
            w_predict = torch.sum(mu_w_sort * w_1divvar_sort, dim=-1) / torch.sum(w_1divvar_sort, dim=-1)
            l_predict = torch.sum(mu_l_sort * l_1divvar_sort, dim=-1) / torch.sum(l_1divvar_sort, dim=-1)
            return loss, {"rewards_chosen": w_predict,
                "rewards_rejected": l_predict,
            }
        # pdb.set_trace()
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        return loss, None, predict


class RewardTrainerForPRM_V4(Trainer):
    """
    V4 splits expert_status into expert_status_w and expert_status_l
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        target_std: float = 0.5,
        max_target_std: Optional[float] = None,
        std_factor: float = 5,
        num_expert: int = 4,
        save_steps: int = 99999,
        output_dir: str = "",
        min_activate_freq: float = 0.15,
        optim_expert: Union[float, int] = 2,
        freq_eps: int = 10
    ):
        """
        [[add]] 
        <train_all_freq> : frequency of optimizing all experts using logsigmoid
        <optim_param> : number or ratio of experts to optimize using logsigmoid
        """
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        assert max_target_std is None or max_target_std > target_std, f"max_target_std({max_target_std}) should be greater than target_std({target_std})"
        assert max_target_std is not None or std_factor > 1, f"std_factor({std_factor}) should be greater than 1"
        if max_target_std is not None:
            print("When max_target_std and std_factor are set simultaneously, std_factor will be neglected")
        else:
            max_target_std = target_std * std_factor

        from math import log
        self.target_std_distribution = torch.linspace(log(target_std), log(max_target_std), num_expert).exp().unsqueeze(0).to("cuda")

        self.count = 0
        assert save_steps > 1
        self.save_steps = save_steps
        self.model_dir = output_dir
        self.div_eps = 1e-3
        self.idx_mat = torch.zeros((1, num_expert)).type(torch.int32)
        self.optim_expert = optim_expert if type(optim_expert) == int else int(num_expert * optim_expert)
        assert min_activate_freq < 0.5*self.optim_expert / num_expert, "min_activate is too large!"
        self.min_activate_freq = min_activate_freq
        self.expert_status_w = torch.full((num_expert,), freq_eps).cuda()
        self.expert_status_l = torch.full((num_expert,), freq_eps).cuda()
        self.iter_total = freq_eps
        self.expert_flag_w = torch.zeros((num_expert,)).bool().cuda()
        self.expert_flag_l = torch.zeros((num_expert,)).bool().cuda()

    def sorted_KL_with_divvar(self, mu, divvar, target_mu, verbose_mode=False):
        """
        mu: [B, N]
        divvar: [B, N]
        target_mu: [B, N]
        calculate D_{KL}(label||predict)
        """
        # clipping !!
        var_ratio = self.target_std_distribution.pow(2) * divvar
        t1 = (target_mu - mu).pow(2) * divvar
        return 0.5 * (var_ratio - (var_ratio + self.div_eps).log() - 1 + t1).sum(1).mean()

    def calculate_target(self, mu_w, logvar_w, mu_l, logvar_l, idx_w, idx_l):
        target_w = mu_w.detach()
        target_l = mu_l.detach()

        # calculate targets for top N=${optim_expert} experts
        mu_w_topk = mu_w[:, :self.optim_expert]
        mu_l_topk = mu_l[:, :self.optim_expert]
        dif_w = mu_w_topk - mu_l[:, [0]]  # [B,oe]
        dif_l = -mu_l_topk + mu_w[:, [0]] 
        target_w[self.idx_mat[:, :self.optim_expert], idx_w[:, :self.optim_expert]] \
            = (mu_w_topk + logvar_w[:, :self.optim_expert].exp()/(dif_w.exp()+1)).detach()
        target_l[self.idx_mat[:, :self.optim_expert], idx_w[:, :self.optim_expert]] \
            = (mu_l_topk - logvar_l[:, :self.optim_expert].exp()/(dif_l.exp()+1)).detach()

        # calculate targets for experts if ${expert_flag}
        # [1,3,2,4,0] [f,f,f,t,f] ==> [f,t,f,f,f]
        bs = mu_w.size(0)
        if self.expert_flag_w.any():    
            bad_idx_w = self.expert_flag_w[idx_w]  # [B,n] 
            mu_w_bad = mu_w[bad_idx_w] # [b*oe]
            dif_bad_w = (mu_w_bad.reshape((bs, -1)) - mu_l[:, [0]]).flatten()
            target_w[bad_idx_w] = (mu_w_bad + logvar_w[bad_idx_w].exp() / (dif_bad_w.exp()+1)).detach()
        if self.expert_flag_l.any():
            bad_idx_l = self.expert_flag_l[idx_l]
            mu_l_bad = mu_l[bad_idx_l]
            dif_bad_l = (-mu_l_bad.reshape((bs, -1)) + mu_w[:, [0]]).flatten()
            target_l[bad_idx_l] = (mu_l_bad - logvar_l[bad_idx_l].exp() / (dif_bad_l.exp()+1)).detach()
        return target_w, target_l

    def sort_and_idx(self, mu, logvar, descending=False):
        bs = mu.size(0)
        if self.idx_mat.size(0) < bs:
            self.idx_mat = torch.zeros_like(mu) + torch.arange(bs).unsqueeze(1).to(mu.device)
            self.idx_mat = self.idx_mat.type(torch.int32)
        mu_sort, idx_sort = torch.sort(mu, descending=descending)
        logvar_sort = logvar[self.idx_mat[:bs], idx_sort]
        return mu_sort, logvar_sort, idx_sort

    def _update_flag(self, idx_w, idx_l):
        bs = idx_w.size(0)
        self.expert_status_w[idx_w[:, :self.optim_expert].reshape(-1)] += 1
        self.expert_status_l[idx_l[:, :self.optim_expert].reshape(-1)] += 1
        self.iter_total += bs
        self.expert_flag_w[self.expert_status_w/self.iter_total < self.min_activate_freq] = True
        self.expert_flag_l[self.expert_status_l/self.iter_total < self.min_activate_freq] = True
        self.expert_flag_w[(self.expert_status_w/self.iter_total > self.min_activate_freq) & self.expert_flag_w] = False
        self.expert_flag_l[(self.expert_status_l/self.iter_total > self.min_activate_freq) & self.expert_flag_l] = True

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self.count += 1
        if self.count % self.save_steps == 0:
            torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")

        mu_w, logvar_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"], no_mle=True)
        mu_l, logvar_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"], no_mle=True)

        mu_w_sort, logvar_w_sort, idx_w = self.sort_and_idx(mu_w, logvar_w, descending=True)
        mu_l_sort, logvar_l_sort, idx_l = self.sort_and_idx(mu_l, logvar_l, descending=False)
        self._update_flag(idx_w, idx_l)

        mu_w_target, mu_l_target = self.calculate_target(mu_w_sort, logvar_w_sort, mu_l_sort, logvar_l_sort, idx_w, idx_l)

        w_1divvar_sort = torch.exp(-logvar_w_sort)
        KL_w = self.sorted_KL_with_divvar(mu_w_sort, w_1divvar_sort, mu_w_target)
        l_1divvar_sort = torch.exp(-logvar_l)
        KL_l = self.sorted_KL_with_divvar(mu_l_sort, l_1divvar_sort, mu_l_target)
        loss = KL_w + KL_l
        print(f"KL_w: {KL_w.item():.3f} KL_l: {KL_l.item():.3f} total: {loss.item(): .3f}")
        print(f"w_freq: {[round(i, 2) for i in (self.expert_status_w/self.iter_total).cpu().tolist()]} l_freq: {[round(i, 2) for i in (self.expert_status_l/self.iter_total).cpu().tolist()]}")
        if return_outputs:
            w_predict = torch.sum(mu_w_sort * w_1divvar_sort, dim=-1) / torch.sum(w_1divvar_sort, dim=-1)
            l_predict = torch.sum(mu_l_sort * l_1divvar_sort, dim=-1) / torch.sum(l_1divvar_sort, dim=-1)
            return loss, {"rewards_chosen": w_predict,
                "rewards_rejected": l_predict,
            }
        # pdb.set_trace()
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        return loss, None, predict


class PairwiseTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        save_steps: int = 99999,
        output_dir: str = ""
    ):
        if type(args) != TrainingArguments:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if type(args) == TrainingArguments:
                if max_length is None:
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.count = 0
        assert save_steps > 1
        self.save_steps = save_steps
        self.model_dir = output_dir
        self.div_eps = 1e-3

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self.count += 1
        if self.count % self.save_steps == 0:
            torch.save(self.model.state_dict(), self.model_dir+f"s{self.count}.pth")

        mu_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
        mu_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"])

        # 0.69315 : E(RM(w) - RM(l)) = 0
        reward_loss = -nn.functional.logsigmoid(mu_w - mu_l).mean()
        self.log({"loss": reward_loss.item(), "chosen": mu_w.mean().item(), "rejected": mu_l.mean().item()})
        if return_outputs:
            return reward_loss, {"rewards_chosen": mu_w,
                "rewards_rejected": mu_l,
            }
        # pdb.set_trace()
        return reward_loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, predict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)
        predict = predict["rewards_chosen"] - predict["rewards_rejected"]
        if "consensus" in inputs:
            label = inputs["consensus"]
        else:
            label = torch.ones_like(predict)
        return loss, predict, label