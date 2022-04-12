from Wardrobe import Wardrobe
#from Recommender import Recommender
#from PIL import Image

#INSTALLATION python3 -m pip install opencv-python

def main():
    w = Wardrobe()
    w.from_csv('./good_matts_wardrobe.csv')
    fit, attr = w.getRandomFit(2)

    for i in fit:
        t = [w.getItem(z) for z in i]
    
    types = []
    for ty in t:
        string = 'NULL' if not ty else ty.type
        types.append(string)


    colors = [col.color for col in t if col]
    print(types, colors, attr)

    # ratings = w.displayFit(fit, attr, '../matts_wardrobe_jpeg')
    # print(ratings)

if __name__ == '__main__':
    main()
