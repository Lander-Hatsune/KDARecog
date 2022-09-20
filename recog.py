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
    if out == 10:
        return '*'
    elif out == 11:
        return '-'
    else:
        return str(out.item())

def getkda(frame:np.ndarray):
    # shape: [height, width, nchannels]
    # kda: [7:21, 1663:1743]
    res = ''
    img = Image.fromarray(frame[7:21, 1663:1743]).convert('L')
    imgarr = np.array(img)

    is_text = False
    for icol in range(imgarr.shape[1]):
        if sum(np.sort(imgarr[:, icol])[-3:]) > 230:
            is_text = True
        else:
            if icol < 9:
                continue
            # icol >= 9
            if is_text:
                is_text = False
                digit = imgarr[:, icol - 9:icol]
                res += ocr1digit(digit)

    res = res.strip('-').strip('*')

    '''
    if res == '4-0-0':
        code = str(np.random.random())[2:8]
        print(code)
        img.save(f'{code}-{res}.png')
    '''
        
    if not res or not re.match(r'\d+-\d+-\d+', res):
        return None
    else:
        return tuple(map(int, res.split('-')))
