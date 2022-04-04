##########
#
# Represents a Wardrobe, which is a collection of Items
#
##########
from Item import Item

class Wardrobe:

    def __init__ (self):
        self.items = []
        self.pks = []

    def addItem(self, clothing_item):
        self.items.append(Item(clothing_item))
        self.pks.append(clothing_item[0])
    
    def getItem(self, primary_key):
        return self.items[self.pks.index(primary_key)]

    def getWardrobe(self):
        return self.items
    
    def __str__ (self):
        string_items = [str(i) for i in self.items]
        return "\n".join(string_items)