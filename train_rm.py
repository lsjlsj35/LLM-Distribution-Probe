import argparse
import json
import random
import torch
from torch.optim import SGD
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from main import probe_dist, load_model


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
    # torch.set_grad_enabled(False)
    # mpath = "/root/model/phi-2"
    # tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    # dp = DistributionProbe(model, tokenizer, max_len=500, max_bs=50, use_model="phi-2")


def generate_data_for_training(inverse_ratio=0.0):
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


def train(dataset_suffix=0.0):
    Q = [
        "What is in the sky during the daytime?",
        "Can you see the moon in the daytime?",
        "Is the sun visible every day in the sky?",
        "What is the brightest object in the daytime sky?",
    ]


def test_train_distributed_model():
    mpath = "/root/model/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    print(tokenizer.pad_token, tokenizer.pad_token_idgit)
    model = load_model(mpath, cuda_list="4,5,6", memory="2GiB")
    
    opt = SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    p = "Instruct: {}\nOutput: {}".format("hello "*10+".", "What do you want to say "*10+"?")
    q = tokenizer([p], return_tensors="pt").to(model.device)
    output = model(**q, return_dict=True).logits.squeeze()
    loss = loss_func(output, q.input_ids.squeeze().to(output.device))
    loss.backward()
    opt.step()
    torch.save(model.state_dict(), "tmp.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inverse-rate", "-i", type=float, default=0.0)
    args = parser.parse_args()
    test_train_distributed_model()
    # preparing_prob_and_rank(args.inverse_rate)
    # generate_data_for_training(
    #     inverse_ratio=args.inverse_rate,
    # )
    # train(
    #     dataset_suffix=str(int(args.inverse_rate * 100))
    # )


