import PIL
import glob
from random import choice
import numpy as np 

def load_image():
    w = choice(glob.glob('.\\mock_img\\*.jpg'))
    print(w)
    with PIL.Image.open(w) as im:
        x = im.resize((224, 224))
    return np.array(x)