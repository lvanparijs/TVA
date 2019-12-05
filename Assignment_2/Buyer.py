class Buyer:
    def __init__(self, index ,limit):
        self.name = index           # Name of Seller
        self.max_limit = limit

    def __str__(self):
        return str(self.name) + "  " + str(self.max_limit)

