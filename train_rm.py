import argparse
import json
import random
import torch
import torch.nn as nn
from torch.optim import SGD
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from main import probe_dist_single_gpu, load_model
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
    parser.add_argument("--base-model-path", type=str, default="/root/exp-modeling/model/RM/phi-2_alpaca-human-")
    parser.add_argument("--dataset-path", "-d", type=str, default="")
    parser.add_argument("--mpath", "-m", type=str, default="0_Exp1")
    parser.add_argument("--calculate_method", "-c", type=str, choices=["mean", "sum"], default="mean")
    parser.add_argument("--logdir", "-l", type=str, default="")
    parser.add_argument("--eos", action="store_true")
    args = parser.parse_args()
    return args


def preparing_prob_and_rank():
    args = get_args()
    dataset_path = args.dataset_path
    eos = args.eos
    model_name = "phi-2"
    probe_dist_single_gpu(
        dataset_path=dataset_path,
        mpath=f"/root/model/{model_name}",
        save_prefix=f"alpaca_human_pref_phi-2_eos_sample",
        use_model=model_name,
        eos=eos
    )


def generate_data_gt_and_get_inverse_rate_NEW():
    # get logit average
    args = get_args()
    logit_avg = []
    with open("qa_status/phi-2_alpaca_human_pref_.jsonl") as f:
        for line in f:
            item = json.loads(line)
            w = item["r"]
            l = item["l"]
            if args.calculate_method == "mean":
                logit_avg.append((sum(w)/len(w), sum(l)/len(l)))
            else:
                logit_avg.append((sum(w), sum(l)))

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
            label = True
        else:
            label = False
        final_data.append({
            "input": q,
            "win": wa,
            "lose": la,
            "consensus": label,
        })
    print("inverse rate:", 1-correct / len(qa_data))
    if args.calculate_method == "mean":
        with open(f"data/phi_2-alpaca_human_pref-Igt.json", "w") as f:
            json.dump(final_data, f, indent=4)
    else:
        with open(f"data/phi_2-alpaca_human_pref-Igt-sum.json", "w") as f:
            json.dump(final_data, f, indent=4)


def generate_data_for_training():
    # get logit average
    args = get_args()
    inverse_ratio = get_args().inverse_rate
    logit_avg = []
    with open("qa_status/phi-2_alpaca_human_pref_.jsonl") as f:
        for line in f:
            item = json.loads(line)
            w = item["r"]
            l = item["l"]
            if args.calculate_method == "mean":
                logit_avg.append((sum(w)/len(w), sum(l)/len(l)))
            else:
                logit_avg.append((sum(w), sum(l)))

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
        flag = 0
        if random.random() < inverse_ratio:
            wa, la = la, wa
            flag += 1
        if wp >= lp:
            final_data.append({
                "input": q,
                "win": wa,
                "lose": la,
                "consensus": flag%2 == 0,
            })
        else:
            final_data.append({
                "input": q,
                "win": la,
                "lose": wa,
                "consensus": flag%2 == 1
            })
    if args.calculate_method == "mean":
        with open(f"data/phi_2-alpaca_human_pref-I{int(100*inverse_ratio)}.json", "w") as f:
            json.dump(final_data, f, indent=4)
    else:
        with open(f"data/phi_2-alpaca_human_pref-I{int(100*inverse_ratio)}-sum.json", "w") as f:
            json.dump(final_data, f, indent=4)


def eval_model_predict_distribution():
    from src.reward_model import RewardModel
    def predict(model, inputs):
        mu_w = model(inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
        mu_l = model(inputs["rejected_input_ids"], inputs["rejected_attention_mask"])
        return mu_w.mean().item(), mu_l.mean().item()
    
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained("/root/model/phi-2")
    model = RewardModel(
        "/root/model/phi-2",
        trust_remote_code=True,
    )
    model.load_state_dict(torch.load(args.base_model_path + args.mpath))
    device = next(model.parameters()).device
    print(device)
    p = "Instruct: {}\nOutput: {}".format("hello "*10+".", "What do you want to say "*10+"?")
    q = tokenizer([p], return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**q)
    print(output)


def train():
    train_pipeline()


def look_up():
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorboard.backend.event_processing import event_accumulator
    logdir = "tensorboard/RM/" + get_args().logdir
    result = glob.glob(logdir)
    print(glob.glob(logdir))
    flag = "y"
    if len(result) > 1:
        flag = input("output?")
    if flag.lower() in ["y", "yes"]:
        Converge = None
        Agree = None
        Disagree = None
        for res in result:
            f = glob.glob(res+"/*")
            name = res.rsplit('/', 1)[1]
            ea = event_accumulator.EventAccumulator(f[0])
            ea.Reload()
            step, acc = [i for i in zip(*[(i.step, i.value) for i in ea.scalars.Items('eval/acc')])]
            agreed_acc = [i.value for i in ea.scalars.Items('eval/agreed_acc')]
            disagreed_acc = [i.value for i in ea.scalars.Items('eval/disagreed_acc')]
            if Converge is None:
                Converge = np.array(acc)
                Agree = np.array(agreed_acc)
                Disagree = np.array(disagreed_acc)
            else:
                Converge += np.array(acc)
                Agree = np.array(agreed_acc)
                Disagree = np.array(disagreed_acc)
            plt.figure()
            plt.plot(step, acc, color="#00F5FF", label="total")
            plt.plot(step, agreed_acc, color="#00FF00", label="agree")
            plt.plot(step, disagreed_acc, color="#FF4500", label="disagree")
            plt.legend()
            plt.savefig("img/"+name+".jpg")
        if Converge is not None:
            L = len(Converge) // 2
            Converge /= len(result)
            Agree /= len(result)
            Disagree /= len(result)
            print("min max converge agree disagree")
            print(np.min(Converge), np.max(Converge), end=" ")
            tmp1 = np.mean(Converge[L:])
            tmp2 = np.mean(Converge[L//2:])
            tmp3 = np.mean(Converge[L//6:])
            print(tmp1, tmp2, end=" ")
            print(np.mean(Agree[L//2:]), np.mean(Disagree[L//2:]))


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
    import inspect
    module = __import__(__name__) 
    all_funcs = [name for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)] 
    if len(sys.argv) > 1:
        func = sys.argv[1]
        if func in UNREACHABLE_LIST:
            raise ValueError(func)
        if func in all_funcs:
            sys.argv.pop(1)
            eval(func)()
        else:
            train()
    else:
        train()
    


