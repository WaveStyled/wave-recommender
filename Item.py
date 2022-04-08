##########
#
# Represents an Item
# Same representation as a row in the SQL Databases
#
##########

class Item:
    attribute_mappings = {"pieceid": 0,
                            "color": 1,
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


                            pieceID INT PRIMARY KEY, \
        COLOR VARCHAR(12), \
        TYPE VARCHAR(5), \
        RECENT_DATE_WORN DATE, \
        TIMES_WORN INT, \
        RATING NUMERIC(3,2) DEFAULT 0.50, \
        OC_FORMAL BOOLEAN, \
        OC_SEMI_FORMAL BOOLEAN, \
        OC_CASUAL BOOLEAN, \
        OC_WORKOUT BOOLEAN, \
        OC_OUTDOORS BOOLEAN, \
        OC_COMFY BOOLEAN, \
        WE_COLD BOOLEAN, \
        WE_HOT BOOLEAN, \
        WE_RAINY BOOLEAN, \
        WE_SNOWY BOOLEAN, \
        WE_AVG_TMP BOOLEAN, \
        DIRTY BOOLEAN

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
