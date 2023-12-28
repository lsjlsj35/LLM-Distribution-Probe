import json
import numpy as np
import torch
from matplotlib import pyplot as plt


def contrast_plot(arr1, arr2, seq, normed=False, name=""):
    if type(arr1) is list:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
    if type(arr1) is torch.Tensor:
        arr1 = arr1.numpy()
        arr2 = arr2.numpy()
    if not normed:
        arr1 += 20
        arr2 += 20
    if len(arr1.shape) > 1:
        arr1 = arr1.squeeze()
    x = np.arange(1, arr1.shape[0]+1)
    plt.plot(x, arr1, "b", label="before")
    plt.plot(x, arr2, "g", label="after")
    plt.xticks(x, seq)
    plt.legend()
    if name:
        plt.savefig("img/"+name+".jpg")
    else:
        plt.savefig("img/contrast.jpg")


fp = "qa_status/glm3_alpaca_human_pref_.jsonl"
R = []
L = []
with open(fp) as f:
    for line in f:
        item = json.loads(line)
        r = np.array(item["r"])
        l = np.array(item["l"])
        R.append(np.mean(r))
        L.append(np.mean(l))
plt.hist(np.array(R)+20, bins=30)
plt.hist(np.array(L)+20, bins=30, alpha=0.5)
plt.savefig("img/tmp3.jpg")


