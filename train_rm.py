import argparse
import json
import random
import torch
import torch.nn as nn
from torch.optim import SGD
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from main import probe_dist, load_model
from src.training_pipeline import train_pipeline


UNREACHABLE_LIST = ["unreachable", "test_mode"]


def unreachable(func):
    UNREACHABLE_LIST.append(func.__name__)
    return func


def test_mode(func):
    def F():
        print("Test start.")
        func()
        print("Test passed!")
    return F


@unreachable
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inverse-rate", "-i", type=float, default=0.0)
    args = parser.parse_args()
    return args


def preparing_prob_and_rank():
    model_name = "phi-2"
    probe_dist(
        max_bs=100,
        cuda_list="6,7",
        memory="3GiB",
        set_host_low_loading=False,
        max_len=500,
        mpath=f"/root/model/{model_name}",
        save_prefix=f"{model_name}_alpaca_human_pref_",
        use_model=model_name
    )


def generate_data_gt_and_get_inverse_rate():
    # get logit average
    logit_avg = []
    with open("qa_status/phi-2_alpaca_human_pref_.jsonl") as f:
        for line in f:
            item = json.loads(line)
            w = item["r"]
            l = item["l"]
            logit_avg.append((sum(w)/len(w), sum(l)/len(l)))

    # get QA data
    with open("/root/dataset/alpaca_farm/alpaca_human_preference.json") as f:
        qa_data = json.load(f)
    assert len(logit_avg) == len(qa_data)

    final_data = []
    correct = 0
    for (wp, lp), item in zip(logit_avg, qa_data):
        q = item["instruction"] + '\n' + item["input"]
        wa = item[f'output_{item["preference"]}']
        la = item[f'output_{3-item["preference"]}']
        if wp >= lp:
            correct += 1
        final_data.append({
            "input": q,
            "win": wa,
            "lose": la,
        })
    print("inverse rate:", 1-correct / len(qa_data))
    with open(f"data/phi_2-alpaca_human_pref-Igt.json", "w") as f:
        json.dump(final_data, f, indent=4)


def generate_data_for_training():
    # get logit average
    inverse_ratio = get_args().inverse_rate
    logit_avg = []
    with open("qa_status/phi-2_alpaca_human_pref_.jsonl") as f:
        for line in f:
            item = json.loads(line)
            w = item["r"]
            l = item["l"]
            logit_avg.append((sum(w)/len(w), sum(l)/len(l)))

    # get QA data
    with open("/root/dataset/alpaca_farm/alpaca_human_preference.json") as f:
        qa_data = json.load(f)
    assert len(logit_avg) == len(qa_data)

    # combine them
    # don't forget that logit_avg = [(win logit, lose logit),...], not [(1,2),...]
    final_data = []
    for (wp, lp), item in zip(logit_avg, qa_data):
        q = item["instruction"] + '\n' + item["input"]
        wa = item[f'output_{item["preference"]}']
        la = item[f'output_{3-item["preference"]}']
        if random.random() < inverse_ratio:
            wa, la = la, wa
        if wp >= lp:
            final_data.append({
                "input": q,
                "win": wa,
                "lose": la,
            })
        else:
            final_data.append({
                "input": q,
                "win": la,
                "lose": wa,
            })
    with open(f"data/phi_2-alpaca_human_pref-I{int(100*inverse_ratio)}.json", "w") as f:
        json.dump(final_data, f, indent=4)


def train():
    train_pipeline()


def probe_distribution_change_after_train():
    Q = [
        "What is in the sky during the daytime?",
        "Can you see the moon in the daytime?",
        "Is the sun visible every day in the sky?",
        "What is the brightest object in the daytime sky?",
    ]


@test_mode
def test_infer_distributed_model():
    class TMP(nn.Module):
        def __init__(self, mpath, cuda_list, memory):
            super().__init__()
            self.model = load_model(mpath, cuda_list=cuda_list, memory=memory)
            self.value_head = nn.Linear(self.model.config.hidden_size, 1, dtype=next(self.model.parameters()).dtype).to(next(self.model.parameters()).device)
            self.one_score=True

        def forward(self, **kwargs):
            sequences = kwargs["input_ids"]
            attention_mask = kwargs["attention_mask"]
            # [batch, seq_len]
            outputs = self.model(**kwargs)
            last_hidden_states = outputs.hidden_states[-1]
            sequence_lengths = torch.max(attention_mask * torch.arange(sequences.size(1), device=attention_mask.device).unsqueeze(0), dim=1)[0]
            sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), sequence_lengths]
            if self.one_score:
                return self.value_head(sequence_hidden_states).squeeze(1)  # ensure shape is (B, )
            return self.value_head(sequence_hidden_states.to(self.value_head.bias.device))
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    mpath = "/root/model/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = TMP(mpath, "4,5,6", memory="2GiB")
    p = "Instruct: {}\nOutput: {}".format("hello "*10+".", "What do you want to say "*10+"?")
    q = tokenizer([p], return_tensors="pt").to(model.device)
    print(q)
    output = model(**q, return_dict=True, output_hidden_states=True)
    print(output)


@test_mode
def test_train_distributed_model():
    mpath = "/root/model/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    print(tokenizer.pad_token, tokenizer.pad_token_id)
    model = load_model(mpath, cuda_list="4,5,6", memory="2GiB")
    opt = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    p = "Instruct: {}\nOutput: {}".format("hello "*10+".", "What do you want to say "*10+"?")
    q = tokenizer([p], return_tensors="pt").to(model.device)
    output = model(**q, return_dict=True).logits.squeeze()
    loss = loss_func(output, q.input_ids.squeeze().to(output.device))
    loss.backward()
    opt.step()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        func = sys.argv.pop(1)
        if func in UNREACHABLE_LIST:
            raise ValueError(func)
        eval(func)()
    else:
        train()
    


