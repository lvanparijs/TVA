from MAS.Seller import Seller
from MAS.Buyer import Buyer
import random

sellers = 2
buyer = 10
rounds = 1
universal_max = 100
penalty = 1

list_sellers = []
list_buyers = []

list_highest_bid = []
list_pay_for_item = []

# Initialize Lists
# Random Number but for testing put it
for i in range(sellers):
    list_sellers.append(Seller(i, 'Chair'))

# Rnadom Number but for testing put it
for i in range(buyer):
    list_buyers.append(Buyer(i ,random.randrange(0, universal_max)))

def pick_second_highest(list_temp):
    index = 0
    for i in list_temp:
        if list_temp[index].max_limit < i.max_limit:
            index = i.name

    del list_temp[index]

    index = 0
    for i in list_temp:
        if list_temp[index].max_limit < i.max_limit:
            index = i.name

    return index

def pure_voting():
    for i in range(rounds):
        for ii in range(sellers):
            index = 0
            for iii in range(buyer):
                if list_buyers[index].max_limit < list_buyers[iii].max_limit and list_buyers[iii].max_limit < universal_max:
                    index = iii
                list_sellers[ii].market_price += list_buyers[iii].max_limit
            list_highest_bid.append(index)
            list_pay_for_item.append(list_buyers[pick_second_highest(list_buyers.copy())])

    for i in range(len(list_highest_bid)):
        print(list_sellers[i].market_price/ buyer)
        print(str(list_highest_bid[i]) + " -- " + str(list_pay_for_item[i]))

pure_voting()
print()
for i in list_buyers:
    print(i)