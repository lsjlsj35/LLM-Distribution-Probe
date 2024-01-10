import json
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def pprint(s, n):
    print(f"[{n}]: ", end='')
    print(s)


def wait_to_be_implemented(func):
    print(f"{func.__name__} has not been implemented yet")
    return func


def wait_to_be_improved(s):
    print("This method has some structural or implementing defects that should be rectified.")
    def F(func):
        return func
    return F


class DistributionProbe:
    def __init__(self, model, tokenizer, max_len=410, max_bs=50, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len  # of tokens
        self.max_bs = max_bs  # for model batched forward
        self.use_model = kwargs.get("use_model", "chatglm2")
        self.with_eos = kwargs.get("eos", False)
        if self.with_eos:
            print("with eos: \n<|endoftext|>")

    def get_prob_and_rank(self, idx, mat, return_log=True):
        """
        idx: [bs,] gt token's index for each query
        mat: [bs, vocab_size] predicted logits for every token 
        """
        mat = F.softmax(mat, dim=-1)
        prob = mat.gather(1, idx.to(torch.int64).unsqueeze(1))
        rank = (mat >= prob).sum(1)
        if return_log:
            return torch.clamp(torch.log(prob.squeeze(1)), min=-20), rank
        return prob.squeeze(1), rank

    @wait_to_be_implemented
    def stream_probing(self, fpath, return_topk=4):
        """
        [fpath] can either be the json file path or the loaded data.
        [return_topk] can either be bool(False means ) or int
        """
        if type(fpath) is list:
            data = fpath
        else:
            with open(fpath) as f:
                data = json.load(f)

    @wait_to_be_implemented
    def get_topk_id_and_prob(self, q, a, k=4, return_log=True):
        pass

    @wait_to_be_improved("""\
    Put the tokenizing procedure out of this function, \
    create a new class(Tokenizer) to support different tokenizing strategy.
    """)
    def probing_data(self, fpath, probe_single=False, save_prefix="alpaca_human_pref_"):
        if type(fpath) is list:
            data = fpath
        else:
            with open(fpath) as f:
                data = json.load(f)
        result = []
        result_for_json = []
        pbar = tqdm(enumerate(data), total=len(data))
        if probe_single:
            for idx, item in pbar:
                r = self._response_single_loop(item)
                pbar.set_description(f'win:{r["status"]["prob_avg"].item():.2f}')
                with open(f"qa_status/{save_prefix}.jsonl", "a") as f:
                    json.dump({
                        "id": idx,
                        "prob": r["status"]["prob"].tolist(),
                        "rank": r["status"]["rank"].tolist(),
                    }, f)
                    f.write('\n')
        else:
            for idx, item in pbar:
                r = self._response_pair_loop(item)
                result.append(r)
                result_for_json.append({
                    "id": idx,
                    "p_win_avg": r["w"]["prob_avg"].item(),
                    "r_win_avg": torch.mean(r["w"]["rank"].to(torch.float16)).item(),
                    "p_lose": r["l"]["prob_avg"].item(),
                    "r_lose_avg": torch.mean(r["l"]["rank"].to(torch.float16)).item(),
                })
                pbar.set_description(f'win:{r["w"]["prob_avg"].item():.2f}  lose:{r["l"]["prob_avg"].item():.2f}')
                with open(f"qa_status/{save_prefix}.jsonl", "a") as f:
                    json.dump({
                        "id": idx,
                        "r": r["w"]["prob"].tolist(),
                        "l": r["l"]["prob"].tolist(),
                        "rr": r["w"]["rank"].tolist(),
                        "lr": r["l"]["rank"].tolist(),
                    }, f)
                    f.write('\n')
            #     if (idx+1) % 1000 == 0:
            #         torch.save(result, f"qa_status/{save_prefix}{idx}.pt")
            #         result = []
            # if (idx+1) % 1000:
            #     torch.save(result, f"qa_status/{save_prefix}{idx}.pt")
            # with open(f"qa_status/{save_prefix}{idx}.json", "w") as f:
            #     json.dump(result_for_json, f, indent=4)

    def _response_single_loop(self, item):
        Q = (item["instruction"] + '\n' +item["input"]).strip()
        A = item["output"]
        # status = {k: v for k, v in zip(["prob", "rank"], self._batch_loop(Q, A))}
        status = {k: v for k, v in zip(["prob", "rank"], self._smart_loop(Q, A))}
        status["prob_avg"] = torch.mean(status["prob"])
        status["output"] = A
        return {
            "status": status,
            "Q": Q,
        }

    def _response_pair_loop(self, item):
        # data preprocessing, should be implemented in a new function.
        if "preference" in item.keys():
            Q = item["instruction"] + '\n' + item["input"]
            prefer = item["preference"]
            assert prefer in [1, 2]
            Awin = item[f"output_{prefer}"]
            Alose = item[f"output_{3-prefer}"]
        else:
            Q = item["input"]
            Awin = item["win"]
            Alose = item["lose"]

        # status_win = {k: v for k, v in zip(["prob", "rank"], self._batch_loop(Q, Awin))}
        status_win = {k: v for k, v in zip(["prob", "rank"], self._smart_loop(Q, Awin))}
        status_win["prob_avg"] = torch.mean(status_win["prob"])
        status_win["output"] = Awin

        # status_lose = {k: v for k, v in zip(["prob", "rank"], self._batch_loop(Q, Alose))}
        status_lose = {k: v for k, v in zip(["prob", "rank"], self._smart_loop(Q, Alose))}
        status_lose["prob_avg"] = torch.mean(status_lose["prob"])
        status_lose["output"] = Alose

        self._greedy_generate(implemented=False)
        status_greedy = {}

        self._sample_generate(implemented=False)
        status_sample = {}
        return {
            "w": status_win,
            "l": status_lose,
            "greedy": status_greedy,
            "sample": status_sample,
            "Q": Q,
        }

    def _sample_generate(self, q=None, top_p=0.85, temperature=0.5, implemented=False):
        if not implemented:
            return None
        ans, prob, rank = "", [0.], [1]
        return ans, prob, rank
    
    def _greedy_generate(self, q=None, implemented=False):
        if not implemented:
            return None
        ans, prob, rank = "", [0.], [1]
        return ans, prob, rank
    
    def _smart_loop(self, q, a):
        if self.use_model == "phi-2":
            pin = "Instruct: {}\nOutput:".format(q)
            pcompletion = "Instruct: {}\nOutput: {}".format(q, a)
            if self.with_eos:
                pcompletion = pcompletion + "\n<|endoftext|>"
            q_seq = self.tokenizer([pin], return_tensors="pt").to("cuda")
            qa_seq = self.tokenizer([pcompletion], return_tensors="pt").to("cuda")
        else:
            raise ValueError(self.use_model)
        with torch.no_grad():
            start = len(q_seq.input_ids[0]) - 1
            mat = self.model(**qa_seq, return_dict=True).logits[0, start:-1, :]  # [1, seq, n]
            gt = qa_seq.input_ids[0, start+1:]
            prob, rank = self.get_prob_and_rank(gt, mat, return_log=True)
        return prob.cpu(), rank.cpu()

    def _batch_loop(self, q, a):
        """
        a batch stands for all data from one Q-A pair
        """
        if self.use_model == "chatglm3":
            q_seq = self.tokenizer.build_chat_input(q, history=[], role="user")["input_ids"].squeeze(0)
            qa_seq = self.tokenizer.build_chat_input(
                "", 
                history=[
                    {"role": "user", "content": q}, 
                    {"role": "assistant", "metadata": "", "content": a}
                ], 
                role="user",
            )["input_ids"][:, :-4].squeeze(0)
        elif self.use_model == "phi-2":
            pin = "Instruct: {}\nOutput:".format(q)
            pcompletion = "Instruct: {}\nOutput: {}".format(q, a)
            q_seq = self.tokenizer([pin], return_tensors="pt")["input_ids"].squeeze(0)
            qa_seq = self.tokenizer([pcompletion], return_tensors="pt")["input_ids"].squeeze(0)
        else:
            pin = "[Round 1]\n\n问：{}\n\n答：".format(q)
            pcompletion = "[Round 1]\n\n问：{}\n\n答：{}".format(q, a)
            q_seq = self.tokenizer([pin], return_tensors="pt")["input_ids"].squeeze(0)
            qa_seq = self.tokenizer([pcompletion], return_tensors="pt")["input_ids"].squeeze(0)
        query_length = q_seq.shape[0]
        total_length = qa_seq.shape[0]
        # assert total_length <= self.max_len
        if total_length > self.max_len:
            total_length = self.max_len
        PROB = []
        RANK = []
        iters = range(query_length, total_length, self.max_bs)
        for batch_start_loc in tqdm(iters, total=len(list(iters)), leave=False):
            bs = min(batch_start_loc+self.max_bs, total_length) - batch_start_loc
            input_ids = torch.zeros(bs, self.max_len, dtype=torch.int)
            attention_mask = torch.zeros_like(input_ids, dtype=torch.int)
            position_ids = torch.zeros_like(input_ids, dtype=torch.int)
            gt = torch.zeros(bs, dtype=torch.int)
            for row in range(bs):
                loc = batch_start_loc + row
                input_ids[row, -loc:] = qa_seq[:loc]
                attention_mask[row, -loc:] = 1
                position_ids[row, -loc:] = torch.arange(loc, dtype=torch.int)
                gt[row] = qa_seq[loc]
            mat = self._inner_batch_loop({
                "input_ids": input_ids.to("cuda"),
                "attention_mask": attention_mask.to("cuda"),
                "position_ids": position_ids.to("cuda")
            })
            prob, rank = self.get_prob_and_rank(gt.to("cuda"), mat, return_log=True)
            PROB.append(prob)
            RANK.append(rank)
        PROB = torch.cat(PROB).cpu()
        RANK = torch.cat(RANK).cpu()
        return PROB, RANK
        
    def _inner_batch_loop(self, tokenized):
        """
        """
        logit_mat = self.model(**tokenized, return_dict=True).logits[:, -1, :]
        return logit_mat


def test_sort_consuming():
    a = torch.rand(52000)
    from time import time
    with torch.no_grad():
        s = time()
        for _ in range(100):
            b = torch.sort(a)
        print(time() - s)
    a = torch.rand(100, 52000)
    with torch.no_grad():
        s = time()
        b = torch.sort(a, dim=-1)
        print(time() - s)


def test_tokenize_length():
    q = "Write a script for a conversation between two people arguing about whether social media has had a positive or negative impact on society."
    a = "Person 1: I think that social media has had a positive effect on society. It\u2019s given us a way to stay connected with friends and family all over the world, and it has allowed us to share ideas and news quickly.\n\nPerson 2: That\u2019s true, but I think the negative effects of social media outweigh the positive. It has made people socially isolated, it has caused increased levels of stress and anxiety and more significantly, it has caused divisions in the way people communicate and engage with each other, both online and in real life. \n\nPerson 1: I disagree with that. I think that social media can also bring people together in meaningful ways. It\u2019s given young people a space to express themselves and find their identity and to share new ideas. \n\nPerson 2: That\u2019s true, but I believe that the negatives are more profound and ever-present. We should think twice before investing so much time and energy in this platform. \n\nPerson 1: What do you think we should do?\n\nPerson 2: We need to be mindful of the amount of time we spend on social media. It should be a tool, not a way of life. We should also be conscious of our language, especially since online communication has little clues of expression, emotion and body language. \n\nPerson 1: I agree, we should"
    mpath = "/root/model/chatglm3-6b"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    pcompletion = tokenizer.build_chat_input(q, )
    idx_completion = tokenizer([pcompletion], return_tensors="pt").to("cuda")
    print(idx_completion["input_ids"])
    print(idx_completion["input_ids"].shape)
    t = tokenizer([pcompletion], return_tensors="pt", padding="max_length", max_length=410).to("cuda")
    print(t)
    

def init():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
    mpath = "/root/chatglm2-6b"
    # mpath = "/root/model/pythia-1.4b"
    model = AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    vocab = tokenizer.get_vocab()


def load_model(mpath, cuda_list="0,1,2", memory="30GiB", host_low_loading=False):
    cuda_list = cuda_list.split(',')
    no_split_module_classes = ["GLMBlock"] if "chatglm" in mpath else ["CodeGenBlock"]
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


def test_chatglm3():
    mpath = "/root/model/chatglm3-6b"
    model = AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    query = "Hello, how are you?"
    
    inputs = tokenizer.build_chat_input(query, history=[], role="user")
    print(inputs)

    res, his = model.chat(tokenizer, query, history=[], max_length=128)
    print(res)
    print(his)

    print(tokenizer.build_chat_input("", history=his, role="user")["input_ids"])
    print(tokenizer.build_chat_input(query, history=his, role="user")["input_ids"])


def test_phi_2():
    # 198 '\n'   50256 <|endoftext|>
    mpath = "/root/model/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    print(tokenizer.eos_token_id)
    model = AutoModelForCausalLM.from_pretrained(mpath,  torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    
    # print(tokenizer.batch_decode([[43993,    25, 23105, 23105, 23105, 23105, 23105,   198, 26410,    25,
    #        383,   749, 10792,  1573,   287,   262,  6827,   318,   366,  5303,
    #       1911,   198, 50256]]))
    inputs = tokenizer("Instruct: hi hi hi hi hi\nOutput:", return_tensors="pt").to("cuda")
    print(inputs)
    outputs = model.generate(**inputs, max_length=200, eos_token_id=tokenizer.eos_token_id)
    print(outputs)
    print(type(outputs))
    text = tokenizer.batch_decode(outputs)[0]
    print(text.split("<|endoftext|>")[0])
    print(text)


def probe_dist_distributed(**kwargs):
    cuda_list = kwargs.get("cuda_list", '0,1,2')
    memory = kwargs.get("memory", '30GiB')
    mpath = kwargs.get("mpath", "/root/chatglm2-6b")
    torch.set_grad_enabled(False)
    # model = AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    model = load_model(mpath, cuda_list, memory, host_low_loading=kwargs.get("set_host_low_loading", False))
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    dp = DistributionProbe(model, tokenizer, **kwargs)
    dp.probing_data("/root/dataset/alpaca_farm/alpaca_human_preference.json", save_prefix=kwargs.get("save_prefix", "tmp"))


def probe_dist_single_gpu(**kwargs):
    mpath = kwargs.get("mpath", "/root/chatglm2-6b")
    dataset_path = kwargs.get("dataset_path", "/root/dataset/alpaca_farm/alpaca_human_preference.json")
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mpath, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    dp = DistributionProbe(model, tokenizer, **kwargs)
    dp.probing_data(dataset_path, save_prefix=kwargs.get("save_prefix", "tmp"))


if __name__ == "__main__":
    # init()
    # test_sort_consuming()
    # test_tokenize_length()
    # test_chatglm3()
    test_phi_2()
    # probe_dist(
    #     max_bs=100,
    #     cuda_list="0,1,2",
    #     memory="6GiB",
    #     set_host_low_loading=True,
    #     max_len=410,
    #     mpath="/root/chatglm2-6b",
    #     save_prefix="glm2_alpaca_human_pref_2k_",
    #     use_model="chatglm2"
    # )

