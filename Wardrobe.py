##########
#
# Represents a Wardrobe, which is a collection of Items
#
##########
from Item import Item

class Wardrobe:
    def __init__ (self):
        self.item_indexes = {"S": [], "T": [], "P": [], "H": [], "O": [], "A": [], "X": []}
        self.items = []
        self.num_items = 0

    def addItem(self, clothing_item): # index 4 is the type and its just checking the last letter
        self.item_indexes[clothing_item[4][-1]].append(self.num_items)
        self.items.append(Item(clothing_item))
        self.num_items+=1
        
    def getItem(self, primary_key):            
        return self.filter(primary_key, "pieceid")[0]
    
    def deleteItem(self, primary_key):  ## This will ruin the getitem calls , perhaps set that val to none?
        ## del self.items[self.itemsself.getItem(primary_key, "pieceid")[0]
        self.num_items-=1

    def getWardrobe(self):
        return self.items
    
    def filter(self, sub_attr, attribute="type"):
        return list(filter(lambda item: item.getAttr(attribute) == sub_attr, self.items))

    def __getitem__ (self, clothing_type):  ## allows for [] notation with the object
        return [self.items[i] for i in self.item_indexes.get(clothing_type)]

    def __str__ (self):
        string_items = [str(i) for i in self.items]
        return "".join(string_items)
    
    def __del__ (self):
        print("Wardrobe eliminated")