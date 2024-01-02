from datasets import load_dataset


__all__ = ["load_and_preprocess"]


def get_config(args):
    dataset_config = DATASET_CONFIG[args.dataset_name]
    for k, v in TOKENIZER_CONFIG.items():
        if k in args.model_name:
            return dataset_config, v
    raise RuntimeError("Unable to find tokenize func.")


def load_and_preprocess(args, tokenizer, split_from_train_ratio=None, shuffle_seed=1):
    dataset_config, tokenize_config = get_config(args)

    train_dataset = load_dataset("json", data_files={"train": dataset_config["path_train"]}, split="train")
    train_dataset = train_dataset.map(
        dataset_config["preprocess_func"],
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "config": tokenize_config},
        remove_columns=dataset_config["remove_columns"]
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["chosen_input_ids"][0]) <= args.reward_config.max_length
        and len(x["rejected_input_ids"][0]) <= args.reward_config.max_length
    ).shuffle(seed=shuffle_seed)

    eval_dataset = None
    if dataset_config["path_eval"] is not None:
        eval_dataset = load_dataset("json", data_files={"eval": dataset_config["path_eval"]}, split="eval")
        eval_dataset = eval_dataset.map(
            dataset_config["preprocess_func"],
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "config": tokenize_config},
            remove_columns=dataset_config["remove_columns"]
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["chosen_input_ids"][0]) <= args.reward_config.max_length
            and len(x["rejected_input_ids"][0]) <= args.reward_config.max_length
        ).shuffle(seed=shuffle_seed)
    else:
        if split_from_train_ratio is not None:
            total = len(train_dataset)
            assert split_from_train_ratio < 0.5
            thre = int(total * split_from_train_ratio)
            eval_dataset = train_dataset.select(range(thre))
            train_dataset = train_dataset.select(range(thre, total))
    print('===')
    print(len(train_dataset))
    return train_dataset, eval_dataset

######################################################################
#                                                                    #
#                      Preprocess Function                           #
#                                                                    #
######################################################################
def Template_Preprocess_Function(examples, tokenizer, config):       #
    """                                                              #
    TEMPLATE function of data preprocessing                          #
    Note that tokenize function is in *config*                       #
    """                                                              #
    new_examples = {                                                 #
        "chosen_input_ids": [],                                      #
        "chosen_attention_mask": [],                                 #
        "rejected_input_ids": [],                                    #
        "rejected_attention_mask": []                                #
    }                                                                #
    func = config["tokenize_func"]  # get tokenize function          #
    for var1, var2 in zip(examples["key1"], examples["key2"]):       #
        # update new_examples                                        #
        question = None                                              #
        answer = None                                                #
        _ = func(question, answer, tokenizer)                        #
        ...                                                          #
    return new_examples                                              #
######################################################################


def preprocess_alpaca_ref(examples, tokenizer, config):
    new_examples = {
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": []
    }
    func = config["tokenize_func"]
    if "consensus" not in examples:
        for q, w, l in zip(examples["input"], examples["win"], examples["lose"]):
            qw = func(q, w, tokenizer)
            ql = func(q, l, tokenizer)
            new_examples["chosen_input_ids"].append(qw["input_ids"])
            new_examples["rejected_input_ids"].append(ql["input_ids"])
            new_examples["chosen_attention_mask"].append(qw["attention_mask"])
            new_examples["rejected_attention_mask"].append(ql["attention_mask"])
        return new_examples
    else:
        new_examples["consensus"] = []
        for q, w, l, con in zip(examples["input"], examples["win"], examples["lose"], examples["consensus"]):
            qw = func(q, w, tokenizer)
            ql = func(q, l, tokenizer)
            new_examples["chosen_input_ids"].append(qw["input_ids"])
            new_examples["rejected_input_ids"].append(ql["input_ids"])
            new_examples["chosen_attention_mask"].append(qw["attention_mask"])
            new_examples["rejected_attention_mask"].append(ql["attention_mask"])
            new_examples["consensus"].append(1 if con else 0)
        return new_examples


######################################################################
#                                                                    #
#                        Tokenize Function                           #
#                                                                    #
######################################################################
def Template_Tokenize_Function(q, a, tokenizer):                     #
    """                                                              #
    TEMPLATE function of tokenization                                #
    input: question, answer, tokenizer                               #
    output: {"input_ids": ..., "attention_mask": ...}                #
    """                                                              #
    return tokenizer([q+a], return_tensors="pt")                     #
######################################################################


def tokenize_phi_2(q, a, tokenizer):
    completion = f"Instruct: {q}\nOutput: {a}"
    return tokenizer([completion], padding="max_length", max_length=512, return_tensors="pt")


######################################################################
#                                                                    #
#                              Config                                #
#                                                                    #
######################################################################
DATASET_CONFIG = {
    "alpaca-human-0": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I0.json",
        "path_eval": None,
        "preprocess_func": preprocess_alpaca_ref,
        "remove_columns": ["input", "win", "lose"],
    },
    "alpaca-human-15": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I15.json",
        "path_eval": None,
        "preprocess_func": preprocess_alpaca_ref,
        "remove_columns": ["input", "win", "lose"],
    },
    "alpaca-human-30": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I30.json",
        "path_eval": None,
        "preprocess_func": preprocess_alpaca_ref,
        "remove_columns": ["input", "win", "lose"],
    },
    "alpaca-human-50": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-I50.json",
        "path_eval": None,
        "preprocess_func": preprocess_alpaca_ref,
        "remove_columns": ["input", "win", "lose"],
    },
    "alpaca-human-gt": {
        "path_train": "/root/exp-modeling/data/phi_2-alpaca_human_pref-Igt.json",
        "path_eval": None,
        "preprocess_func": preprocess_alpaca_ref,
        "remove_columns": ["input", "win", "lose"],
    },
    "chatbot-arena": {
        "path_train": "/root/exp-modeling/data/chatbot_arena.json",
        "path_eval": None,
        "preprocess_func": preprocess_alpaca_ref,
        "remove_columns": ["input", "win", "lose"],
    },
}

TOKENIZER_CONFIG = {
    "phi-2": {
        "tokenize_func": tokenize_phi_2,
    },
}

#############################################################################
# test
def test():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/model/phi-2", trust_remote_code=True)
    dataset_config = DATASET_CONFIG["alpaca-human-0"]
    tokenize_config = TOKENIZER_CONFIG["phi-2"]
    train_dataset = load_dataset("json", data_files={"train": dataset_config["path_train"]}, split="train")
    train_dataset = train_dataset.map(
        dataset_config["preprocess_func"],
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "config": tokenize_config},
        remove_columns=dataset_config["remove_columns"]
    )
    print(train_dataset)
    print(train_dataset[0])
    m = 0
    for item in train_dataset:
        if len(item["chosen_input_ids"][0]) > m:
            m = len(item["chosen_input_ids"][0])
        if len(item["rejected_input_ids"][0]) > m:
            m = len(item["rejected_input_ids"][0])
    print("max_tokens_num:", m)  # 619
    train_dataset = train_dataset.filter(
        lambda x: len(x["chosen_input_ids"][0]) <= 512
        and len(x["rejected_input_ids"][0]) <= 512
    )
    print(train_dataset)
    train_dataset = train_dataset.select(range(1000))
    print(train_dataset)

def test1():
    import json
    t = 0
    c = 0
    d = 0
    m1 = 0
    m2 = 0
    tie = 0
    with open("/root/dataset/chatbot_arena/train.jsonl") as f:
        for line in f:
            t += 1
            item = json.loads(line)
            if len(item["conversation_a"]) == 2 and len(item["conversation_b"]) == 2:
                c += 1
                a = item["conversation_a"]
                b = item["conversation_b"]
                if a[0]["content"].strip() != b[0]["content"].strip():
                    m1 += 1
                if not(
                    a[0]["role"] == "user"
                    and b[0]["role"] == "user"
                    and a[1]["role"] == "assistant"
                    and b[1]["role"] == "assistant"
                ):
                    m2 += 1
                if item["winner"] == "tie":
                    tie += 1
            if len(item["conversation_a"]) <= 4 and len(item["conversation_b"]) <= 4:
                d += 1
    print(t, d, c, m1, m2, tie)

#################################################################################
# data preprocessing
def get_chatbot_arena_single_round():
    import json
    data = []
    with open("/root/dataset/chatbot_arena/train.jsonl") as f:
        for line in f:
            item = json.loads(line)
            if len(item["conversation_a"]) == 2 and len(item["conversation_b"]) == 2 and "tie" not in item["winner"]:
                if item["winner"] == "model_a":
                    win = "conversation_a"
                    lose = "conversation_b"
                elif item["winner"] == "model_b":
                    win = "conversation_b"
                    lose = "conversation_a"
                else:
                    raise KeyError(item["winner"])
                data.append({
                    "input": item["conversation_a"][0]["content"].strip(),
                    "win": item[win][1]["content"].strip(),
                    "lose": item[lose][1]["content"].strip(),
                })
    print(len(data))
    with open("/root/exp-modeling/data/chatbot_arena.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # test()
    # test1()
    get_chatbot_arena_single_round()

