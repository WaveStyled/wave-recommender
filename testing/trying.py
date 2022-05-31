from Wardrobe import Wardrobe

def main():
    w = Wardrobe()
    w.from_csv('./good_matts_wardrobe.csv')
    fit, attr = w.getRandomFit(100)
    ratings, fit, attr = w.displayFit(fit, attr, '../matts_wardrobe_jpeg')
    print(ratings)
    print(fit)
    print(ratings, attr)
    w.outfitToDB(outfits=fit, ratings=ratings, attrs=attr)

    t = []
    for i in fit:
        t = [w.getItem(z) for z in i]

    types = []
    for ty in t:
        string = 'NULL' if not ty else ty.type
        types.append(string)

    colors = [col.color for col in t if col]

    print(types, colors, attr)
    ratings = w.displayFit(fit, attr, '../matts_wardrobe_jpeg')
    print(ratings)


if __name__ == '__main__':
    main()
