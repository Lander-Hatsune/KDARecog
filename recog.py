import re
import torch
import numpy as np
from . import model
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = model.Net(confidence_thr=0.75).to(device)
with pkg_resources.path(__package__, 'Net.pt') as path:
    net.load_state_dict(torch.load(path))
net.eval()

DIGIT_H = 15
DIGIT_W = 9

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def modelio(batch: np.ndarray, debug=None):
    feed = torch.stack([transform(arr) for arr in batch]).to(device)
    out, confidence = net.pred(feed)
    # print(out, confidence)
    # for i, c in enumerate(batch):
    #     Image.fromarray(c).convert('L').save(f'KDARecog/data/{out[i].item()}-{debug}_{np.random.randint(0, 0xFFFFFF):06X}.png')
    res = ""
    for o in out:
        if o == -1:
            res += 'x'
        elif o == 10:
            res += 'd'
        elif o == 11:
            res += 'c'
        else:
            res += str(o.item())
    return res


def getdigits(imgarr: np.ndarray, debug=None):
    # print("getdigits()", debug)
    text_start = None
    on_text = False
    imgarr = imgarr.astype(np.int16)
    batch = []
    for icol in range(imgarr.shape[1]):
        # print(icol, sum(np.sort(imgarr[:, icol])[-2:]))
        # thresh > 300
        if sum(np.sort(imgarr[:, icol])[-2:]) < 320 or \
            icol == imgarr.shape[1] - 1:
            if icol < DIGIT_W:
                continue
            if icol == imgarr.shape[1] - 1:
                icol = imgarr.shape[1]
            # icol >= DIGIT_W
            if on_text:
                on_text = False
                text_start = max(text_start, icol - DIGIT_W)
                text_width = icol - text_start
                digit = np.zeros((imgarr.shape[0], DIGIT_W), dtype=np.uint8)
                digit[:, -text_width:] = imgarr[:, text_start:icol]
                batch.append(digit)
                # print(f"submit [{text_start}:{icol}]")
        else:
            if not on_text:
                # print(f"text start")
                text_start = icol
                on_text = True
    
    if not batch:
        return ""
    
    batch = np.array(batch)
    res = modelio(batch, debug=debug)
    return res
    # res = res.strip('d').strip('x')


RED = np.array([166, 77, 82], dtype=np.float32)
BLUE = np.array([88, 152, 170], dtype=np.float32)
def imgfilter(imgarr, blue):
    color = BLUE if blue else RED
    normalized_color = color / np.linalg.norm(color)
    color_filter = normalized_color / normalized_color.sum()
    filtered_imgarr = np.dot(imgarr, color_filter)
    return np.clip(filtered_imgarr, 0, 255).astype(np.uint8)


def getall(frame: np.ndarray, debug=""):
    ori_arr = frame[0:30, 1550:]

    L_img = Image.fromarray(ori_arr).convert("L")
    L_arr = np.array(L_img)
    ac_arr = np.array(ImageOps.autocontrast(L_img, cutoff=3))
    
    # cut black    
    # print("RPTR") # column sweep
    for rptr in range(1920 - 1550 - 1, DIGIT_W, -1):
        # print(rptr, sum(np.sort(ac_arr[:, rptr])[-3:]))
        if sum(np.sort(ac_arr[:, rptr])[-3:]) > 700:
            break
    rptr += 3

    # print("TPTR") # pix probe @ `time` icon
    for tptr in range(0, 30 - DIGIT_H):
        # print(tptr, np.max(ac_arr[tptr, rptr - 56]))
        if np.max(ac_arr[tptr, rptr - 56]) > 150:
            break
    tptr -= 4

    # ac_arr = ac_arr[tptr:tptr + DIGIT_H, :rptr]
    L_arr = L_arr[tptr:tptr + DIGIT_H, :rptr]
    ori_arr = ori_arr[tptr:tptr + DIGIT_H, :rptr]
    # Image.fromarray(ori_arr).save("KDARecog/data/debug/a-ori.jpg")
    # Image.fromarray(L_arr).save("KDARecog/data/debug/a-L.jpg")
    # Image.fromarray(ac_arr).save("KDARecog/data/debug/a-ac.jpg")

    score_l_a = L_arr[:-324]
    score_r_a = L_arr[-308:-271]
    if np.sort(np.reshape(score_l_a, (-1,)))[-3:].sum() > \
        np.sort(np.reshape(score_r_a, (-1,)))[-3:].sum():
        ally_to_enemy = True
    else:
        ally_to_enemy = False
    # print(ally_to_enemy)

    arrs = [
        L_arr[:, -46:],        # time
        L_arr[:, -125:-76],    # cs
        L_arr[:, -238:-166],   # kda
        imgfilter(ori_arr[:, :-324], ally_to_enemy),            # score l
        imgfilter(ori_arr[:, -308:-271], not ally_to_enemy),    # score r
    ]

    # for i, arr in enumerate(arrs):
    #     print(arr.shape)
    #     ImageOps.autocontrast(
    #         Image.fromarray(arr), cutoff=5).save(f'KDARecog/data/debug/arr_{["T", "CS", "KDA", "SL", "SR"][i]}.jpg')

    time_s, cs_s, kda_s, score_l_s, score_r_s = [
        getdigits(np.array(ImageOps.autocontrast(
            Image.fromarray(arr), cutoff=5)), 
            # debug=debug + "@" + ["T", "CS", "KDA", "SL", "SR"][i],
        ) for i, arr in enumerate(arrs)
    ]
    
    # print(time_s)
    # print(cs_s)
    # print(kda_s)
    # print(score_l_s)
    # print(score_r_s)

    # team score
    if not score_l_s.isdigit() or not score_r_s.isdigit():
        score = None
    else:
        score = (int(score_l_s), int(score_r_s))
        if not ally_to_enemy:
            score = (score[1], score[0])


    # KDA
    if not kda_s or not re.match(r'\d+d\d+d\d+', kda_s):
        kda = None
    else:
        kda = tuple(map(int, kda_s.split('d')))

    # cs
    if not cs_s.isdigit():
        cs = None
    else:
        cs = int(cs_s)

    # time
    if not time_s or not re.match(r'\d\d[xdc]?\d\d', time_s):
        time = None
    else:
        min_ = int(time_s[:2])
        sec_ = int(time_s[-2:])
        time = min_ * 60 + sec_

    return (score, kda, cs, time)


def getscore(frame: np.ndarray):
    return getall(frame)[0]


def getkda(frame: np.ndarray):
    return getall(frame)[1]


def getgametime(frame: np.ndarray):
    return getall(frame)[3]