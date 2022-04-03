##########
#
# Represents an Item
# Same representation as a row in the SQL Databases
#
##########

class Item:
    attribute_mappings = {"pieceid": 0,
                            "r_color": 1,
                            "g_color": 2,
                            "b_color": 3,
                            "type": 4,
                            "recent_date_worn": 5,
                            "times_worn": 6,
                            "rating": 7,
                            "oc_formal": 8,
                            "oc_semi_formal": 9,
                            "oc_casual": 10,
                            "oc_workout": 11,
                            "oc_outdoors": 12,
                            "oc_comfy": 13,
                            "we_cold": 14,
                            "we_hot": 15,
                            "we_rainy": 16,
                            "we_snowy": 17,
                            "we_avg_tmp": 18,
                            "dirty": 19}

    def __init__ (self, row_tuple):
        self.data = tuple(row_tuple)

    def getAttr(self, attr):
        return self.data[Item.attribute_mappings.get(attr)]

    def getPK(self):
        return self.data[0]

    def getColorCodes(self):
        return (self.data[1], self.data[2], self.data[3])

    def getOCs(self):
        return tuple([self.data[i] for i in range(8,14)])

    def getWEs(self):
        return tuple([self.data[i] for i in range(14,19)])

    def __str__ (self):
        return str(self.data)
