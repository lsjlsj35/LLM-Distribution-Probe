import json
import numpy as np
from colorama import Fore as F, Style as S, Back as B
from transformers import AutoTokenizer


class ColorSettor:
    _fmap = {
        0: F.RED,
        1: F.YELLOW,
        2: F.GREEN,
        3: F.RESET,
    }
    _bmap = {
        0: B.BLUE,
        1: B.CYAN,
        2: B.RESET
    }
    _smap = {
        0: S.DIM,
        1: S.NORMAL,
        2: S.BRIGHT
    }
    def __init__(self, verbose=None):
        self.fore = 3
        self.back = 2
        self.style = 1
        if verbose == '1':
            self.init_display1()
        elif verbose == "2":
            self.init_display2()

    def init_display1(self):
        tmp = ["best", "good", "bad", "worst"]
        o = "FORE:"
        for i, c in zip((3,2,1,0), tmp):
            o += self._fmap[i] + c
        o += F.RESET
        o += '\nSTYLE:'
        for i, c in zip((2,1,0), tmp[:3]):
            o += self._smap[i] + c
        o += S.RESET_ALL
        print(o)

    def init_display2(self):
        tmp = ["best", "good", "bad", "worst"]
        o = "FORE: prob  STYLE: rank\n"
        for i in range(3):
            o += self._smap[2-i]
            for i, c in zip((3,2,1,0), tmp):
                o += self._fmap[i] + c
            o += '\n'
        o += F.RESET
        o += S.RESET_ALL
        print(o)

    @property
    def state(self):
        return (self.fore, self.back)

    def _get_fore_and_back(self, status):
        """
        p: 0~20(best)
        r: 1~50000+(worst)
        """
        p, r = status
        output = ""

        if r > 29:
            scolor = 0
        elif r > 1:
            scolor = 1
        else:
            scolor = 2
        if self.style != scolor:
            self.style = scolor
            output += self._smap[scolor]

        if p > 19.8:
            fcolor = 3
        elif p > 18:
            fcolor = 2
        elif p > 12:
            fcolor = 1
        else:
            fcolor = 0
        if self.fore != fcolor:
            self.fore = fcolor
            output += self._fmap[fcolor]
        return output

    def print(self, c, status):
        """
        c: one character
        status: logprob and rank
        """
        print(self._get_fore_and_back(status)+c, end='')

    def seq_print(self, seq, statuss):
        output = ""
        for c, status in zip(seq, statuss):
            if c == "":
                continue
            output += self._get_fore_and_back(status) + c
        print(output)

    def index_seq_print(self, seq, probs, ranks, tokenizer):
        output = ""
        for index, prob, rank in zip(seq, probs, ranks):
            c = tokenizer.tokenizer.convert_id_to_token(index)
            if c == "":
                continue
            output += self._get_fore_and_back((prob, rank)) + c
        output = output.replace('▁', ' ')
        output = output.replace('<0x0A>', '\\n')
        print(output + F.RESET + S.RESET_ALL + B.RESET)
        self.fore, self.style = 3, 1


class Controller:
    def __init__(self, tokenizer, fp, fp_qa, verbose=None):
        self.tokenizer = tokenizer
        self.cs = ColorSettor(verbose=verbose)
        self.status = self._load_status(fp)
        self.completion = self._load_completion(fp_qa)

    def _load_status(self, fp):
        status = []
        with open(fp) as f:
            for line in f:
                status.append(json.loads(line))
        return status

    def _load_completion(self, fp):
        with open(fp) as f:
            c = json.load(f)
        return c

    def _parse_qa_item(self, item):
        """
        prompt will be added.
        """
        if "preference" in item:
            pref = item["preference"]
            q = item["instruction"] + '\n' + item["input"]
            w = item[f"output_{pref}"]
            l = item[f"output_{3-pref}"]
            return {
                "input": q,
                "w": "[Round 1]\n\n问：{}\n\n答：{}".format(q, w),
                "l": "[Round 1]\n\n问：{}\n\n答：{}".format(q, l)
            }
        else:
            q = item["input"]
            w = item["win"]
            l = item["lose"]
            return {
                "input": q,
                "w": "Instruct: {}\nOutput: {}\n<|endoftext|>".format(q, w),
                "l": "Instruct: {}\nOutput: {}\n<|endoftext|>".format(q, l)
            }

    def _display_a(self, pin, pcompletion, p_seq, r_seq, p_normalized=False):
        """
        pin: prompted input
        pcompletion: prompted completion (with input)
        p_seq: probability sequence
        r_seq: rank sequence
        p_normalized: whether p_seq has been normalized.
        """
        q_seq = self.tokenizer([pin], return_tensors="pt")["input_ids"].squeeze(0)
        qa_seq = self.tokenizer([pcompletion], return_tensors="pt")["input_ids"].squeeze(0)
        a_seq = qa_seq[len(q_seq):].tolist()
        if not p_normalized:
            p_seq = [20+i for i in p_seq]
        self.cs.index_seq_print(a_seq, p_seq, r_seq, self.tokenizer)

    def _display_one_pair(self, qa_item, status_item):
        """
        status_item: id | r | l | rr | lr
        parsed qa_item: (not prompted)input | w | l
        """
        qa_item = self._parse_qa_item(qa_item)
        print(f"No.{status_item['id'] + 1}\nQ:\n" + qa_item["input"])
        qa_item["input"] = "[Round 1]\n\n问：{}\n\n答：".format(qa_item["input"])
        print("[[better response]]:")
        self._display_a(qa_item["input"], qa_item["w"], status_item["r"], status_item["rr"])
        print("[[worse response]]:")
        self._display_a(qa_item["input"], qa_item["l"], status_item["l"], status_item["lr"])    

    def display(self, n):
        for idx, (qa_item, status_item) in enumerate(zip(self.completion, self.status)):
            if idx >= n:
                break
            print('-' * 40)
            self._display_one_pair(qa_item, status_item)


def analyze(n=10):
    fpath = "qa_status/glm2_alpaca_human_pref_.jsonl"
    fp_qa = "/root/dataset/alpaca_farm/alpaca_human_preference.json"
    mpath = "/root/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    controller = Controller(tokenizer, fpath, fp_qa)
    controller.display(n)


def test():
    q = "Append the following sentence to the end of the input.\nThe house saw strange events that night."
    a = "The house saw strange events that night, and something was lurking in the shadows."
    cs = ColorSettor()
    mpath = "/root/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    pin = "[Round 1]\n\n问：{}\n\n答：".format(q)
    pcompletion = "[Round 1]\n\n问：{}\n\n答：{}".format(q, a)
    q_seq = tokenizer([pin], return_tensors="pt")["input_ids"].squeeze(0)
    qa_seq = tokenizer([pcompletion], return_tensors="pt")["input_ids"].squeeze(0)
    a_seq = qa_seq[len(q_seq):].tolist()
    p_seq = [-13.078125, -0.0212249755859375, -0.004405975341796875, 0.0, -0.0014657974243164062, 0.0, 0.0, -0.1346435546875, -0.020721435546875, -16.640625, -2.408203125, -11.46875, 0.0, -1.0380859375, 0.0, -0.09844970703125, -0.0009775161743164062]
    p_seq = [20+i for i in p_seq]
    r_seq = [59, 1, 1, 1, 1, 1, 1, 1, 1, 10086, 4, 75, 1, 2, 1, 1, 1]
    print("Q:")
    print(q)
    print("A:")
    cs.index_seq_print(a_seq, p_seq, r_seq, tokenizer)

if __name__ == "__main__":
    analyze(500)
    # test()