##########
#
# Represents a Wardrobe, which is a collection of Items
#
##########
import Item

class Wardrobe:

    def __init__ (self):
        self.items = []

    def addItem(self, clothing_item):
        self.items.append(Item(clothing_item))
    
    def getItem(self, primary_key):
        pks = [self.items.getPK() for _ in self.items]
        return self.items[pks.index(primary_key)]

    def getWardrobe(self):
        return self.items