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
    return str(out.item()) if out != 10 else '*'

def getkda(frame:np.ndarray):
    # shape: [height, width, nchannels]
    # kda: [7:21, 1666:1743]
    res = ''
    img = Image.fromarray(frame[7:21, 1666:1743]).convert('L')
    imgarr = np.array(img)

    is_text = False
    text_start = -1
    for icol in range(imgarr.shape[1]):
        if sum(np.sort(imgarr[:, icol])[-3:]) > 250:
            is_text = True
            if text_start == -1:
                text_start = icol
        else:
            if is_text:
                is_text = False
                if icol - text_start >= 6: # minimum width of '1'
                    digit = imgarr[:, max(0, icol - 9):icol]
                    res += ocr1digit(digit)
                else:
                    res += '-'
                text_start = -1
    return res
