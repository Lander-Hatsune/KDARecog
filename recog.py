import re
import torch
import numpy as np
from . import model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = model.Net().to(device)
with pkg_resources.path(__package__, 'Net.pt') as path:
    net.load_state_dict(torch.load(path))


def ocr1digit(digit: np.ndarray):
    feed = torch.tensor(digit / 255, dtype=torch.float).unsqueeze(0).to(device)
    out = net(feed).max(1)[1]
    # if True: #out.item() in [4, 7, 8, 9, 11]:
    #     (Image.fromarray(digit)
    #      .convert('L')
    #      .save(f'KDARecog/data/{out.item()}-{np.random.randint(0, 0xFFFF):04X}.png'))
    if out == 10:
        return '*'
    elif out == 11:
        return '-'
    elif out == 12:
        return ':'
    else:
        return str(out.item())


def getdigits(imgarr: np.ndarray):
    text_start = None
    on_text = False
    res = ''
    imgarr = imgarr.astype(np.int16)
    for icol in range(imgarr.shape[1]):
        if sum(np.sort(imgarr[:, icol])[-2:]) > 235:
            if not on_text:
                text_start = icol
                on_text = True
        else:
            if icol < 9:
                continue
            # icol >= 9
            if on_text:
                on_text = False
                text_start = max(text_start, icol - 9)
                text_width = icol - text_start
                digit = np.zeros((imgarr.shape[0], 9))
                digit[:, -text_width:] = imgarr[:, text_start:icol]
                res += ocr1digit(digit)

    res = res.strip('-').strip('*')
    return res


def cutBlack(slice: np.ndarray):
    rightPtr = slice.shape[1] - 1
    channelSum = slice.sum(2)
    while sum(np.sort(channelSum[:, rightPtr])[-3:]) < 200:
        rightPtr -= 1
        if rightPtr <= 1500:
            break
    topPtr = 0
    while sum(np.sort(channelSum[topPtr, :])[-3:]) < 200:
        topPtr += 1
        if topPtr >= 500:
            break
    return slice[topPtr:, :rightPtr]


def getkda(frame: np.ndarray):
    # shape: [height, width, nchannels]
    # kda: [7:21, -254:-160]
    slice = cutBlack(frame)
    img = ImageOps.autocontrast(
        Image.fromarray(slice[7:21, -254:-160]).convert('L'), cutoff=10)
    imgarr = np.array(img)
    res = getdigits(imgarr)

    scoreboard = _getscoreboard(slice)

    # print(res)
    if not res or not re.match(r'\d+-\d+-\d+', res):
        return None
    else:
        return tuple(map(int, res.split('-'))), scoreboard

def getscoreboard(frame: np.ndarray):
    slice = cutBlack(frame)
    return _getscoreboard(slice)


def _getscoreboard(slice: np.array):
    img0 = Image.fromarray(slice[7:21, -362:-339]).convert('L')
    img1 = Image.fromarray(slice[7:21, -324:-301]).convert('L')

    # red is dimmer than blue
    if np.sort(np.reshape(img0, (-1,)))[-3:].sum() > \
        np.sort(np.reshape(img1, (-1,)))[-3:].sum():
        ally_to_enemy = True
    else:
        ally_to_enemy = False

    img0_ac = ImageOps.autocontrast(img0, cutoff=10)
    img1_ac = ImageOps.autocontrast(img1, cutoff=10)
    res0 = getdigits(np.array(img0_ac))
    res1 = getdigits(np.array(img1_ac))
    print((res0, res1) if ally_to_enemy else (res1, res0))
    if not res0.isdigit() or not res1.isdigit():
        return None
    return (res0, res1) if ally_to_enemy else (res1, res0)


def getgametime(frame: np.ndarray):
    # shape: [height, width, nchannels]
    # gametime: [7:21, -64:]
    slice = cutBlack(frame)
    img = ImageOps.autocontrast(
        Image.fromarray(slice[7:21, -64:]).convert("L"), cutoff=10)
    imgarr = np.array(img)
    res = getdigits(imgarr)
    # img.save(f"KDARecog/data/{res.replace(':', 'c').replace('*', 'x')}-{np.random.randint(0, 0xFFFF):04X}.png")

    if not res or not re.match(r'\d\d[\*-:]?\d\d', res):
        return None
    else:
        min_ = int(res[:2])
        sec_ = int(res[-2:])
        return min_ * 60 + sec_
