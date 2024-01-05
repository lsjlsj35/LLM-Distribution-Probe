"""
sample results for Phi-2
Note that the probing result is not strictly the same with sample result for they don't share the same format
probe format Q: |Instruct: {q}\nOutput:| 
probe format A: |Instruct: {q}\nOutput: {A}| <-- there is a space between 'Output' and Answer
model output(sample format, searching methods such as beam search will maximize the logit based on this form, 
                            with some changes like repetition penalty):
                |Instruct: {q}\nOutput:{A}|  <-- In usual, A starts with ' ' and ends with '\n<|endoftext|>'
    I'll strip the prefixing space and suffixing '\n<|endoftext|>'
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Sampler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def sample_sentence_beam_and_topk(self, q):
        p = f"Instruct: {q}\nOutput:"
        length = len(p)
        p = self.tokenizer(p, return_tensors="pt").to(self.model.device)
        result1 = self.model.generate(**p, 
                                      return_dict_in_generate=True, 
                                      max_length=512,
                                      eos_token_id=tok.eos_token_id, 
                                      pad_token_id=50256,
                                      num_beams=4
                                      )
        result2 = self.model.generate(**p, 
                                      return_dict_in_generate=True, 
                                      max_length=512,
                                      eos_token_id=tok.eos_token_id, 
                                      pad_token_id=50256,  # eos_token_id
                                      top_p=0.85,
                                      temperature=1.2,
                                      do_sample=True,
                                      )
        result1 = tok.batch_decode(result1.sequences)[0][length:]
        result2 = tok.batch_decode(result2.sequences)[0][length:]
        for _ in range(3):
            if result2 == result1:
                result2 = self.model.generate(**p, 
                                            return_dict_in_generate=True, 
                                            max_length=512,
                                            eos_token_id=tok.eos_token_id, 
                                            pad_token_id=50256,  # eos_token_id
                                            top_p=0.85,
                                            temperature=2.,
                                            do_sample=True,
                                            )
                result2 = tok.batch_decode(result2.sequences)[0][length:]
            else: break
        if result1.endswith("<|endoftext|>"):
            result1 = result1[:-13]
        if result2.endswith("<|endoftext|>"):
            result2 = result2[:-13]
        return result1.strip(), result2.strip()
    

def sample_anthropic_human_pref_phi_2(RANK=0):
    import json
    from tqdm import tqdm
    mpath = "/root/model/phi-2"
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    tok.pad_token_id = 50256
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    s = Sampler(model, tok)

    with open("/root/exp-modeling/data/phi_2-alpaca_human_pref-I0.json") as f:
        data = json.load(f)
    DATA = []
    pbar = tqdm(data[RANK::4])
    for item in pbar:
        result1, result2 = s.sample_sentence_beam_and_topk(item["input"])
        DATA.append({"input": item["input"], "win": result1, "lose": result2})
        print("input:\n" + item["input"])
        print("win:\n" + result1)
        print("lose:\n" + result2)
    with open("/root/exp-modeling/data/tmp_"+str(RANK)+".json", "w") as f:
        json.dump(DATA, f, indent=4)


def combine(total=4):
    import json
    data = []
    for i in range(total):
        with open(f"/root/exp-modeling/data/tmp_{i}.json") as f:
            data.extend(json.load(f))
    print(len(data))
    with open("/root/exp-modeling/data/alpaca_human_pref_phi_2_sample.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # sample_anthropic_human_pref_phi_2(RANK=3)
    combine()