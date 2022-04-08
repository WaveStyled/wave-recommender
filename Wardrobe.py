##########
#
# Represents a Wardrobe, which is a collection of Items
#
##########
from Item import Item
import pandas as pd

class Wardrobe:

    def __init__ (self):
        dt = pd.DataFrame(columns=['pieceid', "color", "type", "recent_date_worn", "times_worn", "rating", 
                                    "oc_formal", "oc_semi_formal", "oc_casual", "oc_workout", "oc_outdoors",
                                    "oc_comfy", "we_cold", "we_hot", "we_rainy", "we_snowy", "we_typical", "dirty"])

    def addItem(self, clothing_item : tuple): # index 4 is the type and its just checking the last letter
        self.dt.loc[self.dt.shape[0]] = clothing_item
        
    def getItem(self, primary_key : int) -> tuple:          
        return list(self.df.loc[self.df['pieceid'] == primary_key].to_records(index=False))[0]
    
    def getItemasItem(self, primary_key : int) -> Item:
        return Item(self.getItem(primary_key))

    def deleteItem(self, primary_key):  ## This will ruin the getitem calls , perhaps set that val to none?
        ## del self.items[self.itemsself.getItem(primary_key, "pieceid")[0]
        self.num_items-=1

    def getWardrobe(self):
        return self.items
    
    def filter(self, sub_attr, attribute="type"):
        return list(filter(lambda item: item.getAttr(attribute) == sub_attr, self.items))

    def getItemObj(self, id : int) -> Item:
        pass
    
    def getdf(self) -> pd.DataFrame:
        return self.dt

    def __getitem__ (self, clothing_type):  ## allows for [] notation with the object
        return [self.items[i] for i in self.item_indexes.get(clothing_type)]

    def __str__ (self):
        string_items = [str(i) for i in self.items]
        return "".join(string_items)

    def __del__ (self):
        print("Wardrobe eliminated")