import torch
import numpy as np
from KDARecog.model import Net
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip

STRIDE = 1
vid = VideoFileClip('../temp/ziyu-20220713/vid3.mp4').subclip(168, 537)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
net.load_state_dict(torch.load('Net.pt'))

def ocr1digit_(digit:np.ndarray):
    feed = torch.tensor(digit / 255, dtype=torch.float).unsqueeze(0).to(device)
    out = net(feed).max(1)[1]
    return str(out.item()) if out != 10 else 'n'

def getkda_(frame:np.ndarray):
    # shape: [height, width, nchannels]
    # kda: [7:21, 1666:1743]
    kdastr = ''
    img = Image.fromarray(frame[7:21, 1666:1743]).convert('L')
    imgarr = np.array(img)

    digits = []
    
    is_text = False
    text_start = -1
    for icol in range(imgarr.shape[1]):
        if sum(np.sort(imgarr[:, icol])[-3:]) > 230:
            is_text = True
            if text_start == -1:
                text_start = icol
        else:
            if is_text:
                is_text = False
                digit = imgarr[:, max(0, icol - 9):icol]
                digits.append(digit)
                res = ocr1digit_(digit)
                kdastr += res
                text_start = -1
    print(kdastr)
    for digit, res in zip(digits, kdastr):
        Image.fromarray(digit).save(
            f'data/{res}/{str(np.random.random())[2:8]}.png')

for t in range(0, vid.duration, STRIDE):
    getkda_(vid.get_frame(t))
