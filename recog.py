import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import Net
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
model.load_state_dict(torch.load('Net.pt'))

def ocr1digit(digit:np.ndarray):
    assert(digit.shape == (14, 9))
    feed = torch.tensor(digit / 256, dtype=torch.float).unsqueeze(0).to(device)
    out = model(feed).max(1)[1]
    return str(out.item()) if out != 10 else '-'

def getkda(frame:np.ndarray):
    # shape: [height, width, nchannels]
    # kda: [7:21, 1667:1743]
    res = ''
    img = Image.fromarray(frame[7:21, 1667:1743]).convert('L')
    imgarr = np.array(img)
    
    is_text = False
    for icol in range(imgarr.shape[1]):
        if sum(np.sort(imgarr[:, icol])[-3:]) > 300:
            is_text = True
        else:
            if is_text:
                is_text = False
                res += ocr1digit(imgarr[:, icol - 9:icol])

    return res
