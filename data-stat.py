import os
from PIL import Image
import numpy as np

path = "data-13/train"

stats = np.zeros((13, 14, 9))

for i in range(13):
    cnt = 0
    cum = np.zeros((14, 9))
    for filename in os.listdir(f"{path}/{i}/"):
        if not os.path.isfile(f"{path}/{i}/{filename}"):
            continue
        cnt += 1
        
        ipath = f"{path}/{i}/{filename}"
        ilabel = i

        imgarr = np.array(Image.open(ipath))
        cum += imgarr
    cum = cum / cnt
    Image.fromarray(cum).convert("L").save(f"{ilabel}.cum.png")
    stats[ilabel] = cum

with open("stats.npy", "wb") as f:
    np.save(f, stats)

# validating

path = "data-13/validate"

with open("stats.npy", "rb") as f:
    stats = np.load(f)

for i in range(13):
    success_cnt = 0
    cnt = 0
    print(f"label {i}:")
    for filename in os.listdir(f"{path}/{i}/"):
        if not os.path.isfile(f"{path}/{i}/{filename}"):
            continue
        cnt += 1
        
        ipath = f"{path}/{i}/{filename}"
        ilabel = i

        imgarr = np.array(Image.open(ipath))
        diff = (stats - imgarr).reshape(13, -1)
        normdiff = np.linalg.norm(diff, ord=1, axis=1)
        res = np.argmin(normdiff)

        print(f"{filename}, {res}, {normdiff}")

        if res == ilabel:
            success_cnt += 1

    print(f"label {i}: success {success_cnt} / {cnt}")






