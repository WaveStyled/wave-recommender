import requests  ## DUMMY TESTER CLASS

import requests
from random import randint
import pandas as pd
from Wardrobe import Wardrobe
#from PIL import Image

def main():
    w = Wardrobe()
    w.from_csv('./good_matts_wardrobe.csv')
    fit, attr = w.getRandomFit()
    out = [(f, a) for f, a in zip(fit, attr)]
    w.displayFit(fit[0])
    print(w)
    

if __name__ == '__main__':
    main()


