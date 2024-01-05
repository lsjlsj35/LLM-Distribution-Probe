import json
import numpy as np
import torch
from matplotlib import pyplot as plt


# def contrast_plot(arr1, arr2, seq, normed=False, name=""):
#     if type(arr1) is list:
#         arr1 = np.array(arr1)
#         arr2 = np.array(arr2)
#     if type(arr1) is torch.Tensor:
#         arr1 = arr1.numpy()
#         arr2 = arr2.numpy()
#     if not normed:
#         arr1 += 20
#         arr2 += 20
#     if len(arr1.shape) > 1:
#         arr1 = arr1.squeeze()
#     x = np.arange(1, arr1.shape[0]+1)
#     plt.plot(x, arr1, "b", label="before")
#     plt.plot(x, arr2, "g", label="after")
#     plt.xticks(x, seq)
#     plt.legend()
#     if name:
#         plt.savefig("img/"+name+".jpg")
#     else:
#         plt.savefig("img/contrast.jpg")


# fp = "qa_status/glm3_alpaca_human_pref_.jsonl"
# R = []
# L = []
# with open(fp) as f:
#     for line in f:
#         item = json.loads(line)
#         r = np.array(item["r"])
#         l = np.array(item["l"])
#         R.append(np.mean(r))
#         L.append(np.mean(l))
# plt.hist(np.array(R)+20, bins=30)
# plt.hist(np.array(L)+20, bins=30, alpha=0.5)
# plt.savefig("img/tmp3.jpg")

X = [1, 0.9, 0.75, 0.56, 0.48, 0]
# min
y1 = [42, 41.5, 42.8, 48, 52, 56.6]
# max
y2 = [52, 48.8, 49.5, 60, 60, 59.5]
# converge, 236 up; 45 down 
y3 = [48, 45, 45, 57, 58.5, 58.7]
# agreed
y4 = [70, 65, 67, 41, 37, 31]
# disagreed
y5 = [17, 25, 24, 70, 82, 85]
plt.plot(X, y1, color="#8EE5EE", label="min")
plt.plot(X, y2, color="#00C5CD", label="max")
plt.plot(X, y3, color="#00F5FF", label="converge")
plt.plot(X, y4, color="#00FF00", label="agree")
plt.plot(X, y5, color="#FF4500", label="disagree")
plt.quiver(X[1:], y3[1:], [0, 0, 0, 0, 0], [4, 4, -4, -4, 4], color=[(0, 1, 0, 0.5), (0, 1, 0, 0.5), (1, 0, 0, 0.5), (1, 0, 0, 0.5), (0, 1, 0, 0.5)])
plt.legend()
plt.savefig("img/result.jpg")

