import re
import torch
import numpy as np
from . import model
from PIL import Image
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = model.Net().to(device)
with pkg_resources.path(__package__, 'Net.pt') as path:
    net.load_state_dict(torch.load(path))

def ocr1digit(digit:np.ndarray):
    feed = torch.tensor(digit / 255, dtype=torch.float).unsqueeze(0).to(device)
    out = net(feed).max(1)[1]
    # if True: #out.item() in [4, 7, 8, 9, 11]:
    #     (Image.fromarray(digit)
    #      .convert('L')
    #      .save(f'data-predict/{out.item()}/{np.random.randint(0, 0xFFFF):04X}.png'))
    if out == 10:
        return '*'
    elif out == 11:
        return '-'
    elif out == 12:
        return ':'
    else:
        return str(out.item())

def getdigits(imgarr):
    text_start = None
    on_text = False
    res = ''
    for icol in range(imgarr.shape[1]):
        if sum(np.sort(imgarr[:, icol])[-3:]) > 350:
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

def getkda(frame:np.ndarray):
    # shape: [height, width, nchannels]
    # kda: [7:21, 1663:1743] +[7:21, 1653:1743]+
    img = Image.fromarray(frame[7:21, 1663:1743]).convert('L')
    imgarr = np.array(img)
    res = getdigits(imgarr)

    if not res or not re.match(r'\d+-\d+-\d+', res):
        return None
    else:
        return tuple(map(int, res.split('-')))

def getgametime(frame:np.ndarray):
    # shape: [height, width, nchannels]
    # gametime: [7:21, 1856:1904]
    img = Image.fromarray(frame[7:21, 1854:1904]).convert('L')
    imgarr = np.array(img)
    res = getdigits(imgarr)

    if not res or not re.match(r'\d\d[\*-:]?\d\d', res):
        return None
    else:
        min_ = int(res[:2])
        sec_ = int(res[-2:])
        return min_ * 60 + sec_
                
    
