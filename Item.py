##########
#
# Represents an Item
# Same representation as a row in the SQL Databases
#
# DEPRECATED AS OF CURRENT PANDAS OVERHAUL
##########

class Item:
    attribute_mappings = {"pieceid": 0,
                            "color": 1,
                            "type": 2,
                            "recent_date_worn": 3,
                            "times_worn": 4,
                            "rating": 5,
                            "oc_formal": 6,
                            "oc_semi_formal": 7,
                            "oc_casual": 8,
                            "oc_workout": 9,
                            "oc_outdoors": 10,
                            "oc_comfy": 11,
                            "we_cold": 12,
                            "we_hot": 13,
                            "we_rainy": 14,
                            "we_snowy": 15,
                            "we_typical": 16,
                            "dirty": 17 }

    def __init__ (self, row_tuple):
        self.data = tuple(row_tuple)

    def getAttr(self, attr : str):
        return self.data[Item.attribute_mappings.get(attr)]

    def getPK(self) -> int:
        return self.data[0]

    def getColorStr(self) -> str:
        return self.data[1]

    def getOCs(self):
        return tuple([self.data[i] for i in range(6,12)])

    def getWEs(self):
        return tuple([self.data[i] for i in range(12,17)])

    def __str__ (self):
        return str(self.data)
