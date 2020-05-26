# Will Dolan
# Spring 2017
# Stocks Adaboost Simulator
# -*- coding: utf-8 -*-

import sys
import random
import math
import ast

## This program takes a json file describing the set of attributes and weights
## resulting from an adaboost multilearner. It then applies these attributes to a
## csv test file with the same attributes and simulates purchasing on the date
## if the learners predict a '1'.

## position holding: the inventory structure will keep track of the purchase
## price/date and will sell an instance when the current price equals or
## exceeds 1.15x times the price bought.

## at the end the total cash and worth of the assets will be calculated.

## usage: python simulation.py buy_learner.json sell_learner.json test.csv 

##############################################################################
attr_buffer = 2
INIT_CASH = 100000

class Inventory:
	def __init__(self, cash):
		self.cash = cash
		self.stocks = []

	def buy_stock(self, stock):
		if self.cash > stock.price:
			self.stocks.append(stock)
			self.cash = self.cash - stock.price

	def sell_stocks(self, price):
		self.cash += len(self.stocks) * price
		self.stocks = []

	def sell_stock(self, date, current_price):
		for i, o in enumerate(self.stocks):
		    if o.date == date:
		        del self.stocks[i]
		        self.cash += current_price


		# if len(self.stocks) > 0:
		# 	stocks_worth = sum(map(lambda x: x.price, self.stocks))
		# 	self.cash += stocks_worth
		# 	self.stocks = []


class stock:
	def __init__(self, date, price):
		self.date = date
		self.price = price
	def __str__(self):
		print (self.date, self.price)



### File string to float values conversion
def clean_data(data):
	keys = data[0].split(',')
	cleaned_data = []
	for line in data[1:]:
		vals = line.split(',')

		# attr_buffer, globally specified, is how many attr at start to ignore
		vals[attr_buffer:] = map(lambda x: float(x), vals[attr_buffer:])
		new_item = {}
		for i in range(0,len(vals)):
			new_item[keys[i]] = vals[i]
		cleaned_data.append(new_item)
	return cleaned_data

### Given a decider with threshold and pos direction and an instance, 
### predict the instance's label using its attribute.
def predict_y(decider, instance):
	attr = decider['attr_name']
	instance_attr = instance[attr]

	# if the classifier declares values less than threshold as positive:
	if decider['dir'] == 'lt':
		if instance_attr < decider['threshold']:
			return 1
		else:
			return -1

	# else if the classifier declares values greater than thresh. as pos.:
	else:
		if instance_attr > decider['threshold']:
			return 1
		else:
			return -1

## open learners
## buyer first
try: learnerfile = sys.argv[1]
except(IndexError): sys.exit(1)
f = open(learnerfile)
Buy_Learners = []
for line in f.read().splitlines():
	Buy_Learners.append(ast.literal_eval(line))
f.close()

## sell learner second
try: learnerfile = sys.argv[2]
except(IndexError): sys.exit(1)
f = open(learnerfile)
Sell_Learners = []
for line in f.read().splitlines():
	Sell_Learners.append(ast.literal_eval(line))
f.close()

### Applying to Test Set (third)
try: testfile = sys.argv[3]
except(IndexError): sys.exit(1)

print "\nApplying learners to test data.\n"
f2 = open(testfile)
test_input = f2.readlines()
f2.close()

Test_Set = clean_data(test_input)
Stock_Inventory = Inventory(INIT_CASH)
print "Simulation beginning with ${0}.".format(Stock_Inventory.cash)

for stock_instance in Test_Set:							# for each day represented:
	buy_neg_vote = buy_pos_vote = sell_neg_vote = sell_pos_vote = 0
	
	for j in range(0, len(Buy_Learners)):
		## see if buy
		pred_y_buy = predict_y(Buy_Learners[j], stock_instance)   # get a base learner's buy pred
		if pred_y_buy == 1: buy_pos_vote += Buy_Learners[j]['weight']
		else: buy_neg_vote += Buy_Learners[j]['weight']

	if buy_pos_vote > buy_neg_vote and Stock_Inventory.cash * .05 > stock_instance['Close']: 
		buy_cash_amt = buy_pos_vote * 0.05 * Stock_Inventory.cash
		print 'Buying ${0:.2f} of {1} stock at ${2:.2f} on {3}'.format(buy_cash_amt, stock_instance['Symbol'], stock_instance['Close'], stock_instance['Date'])
		print 'Cash left: ${0:.2f}'.format(Stock_Inventory.cash)
	else: buy_cash_amt = 0	

	while buy_cash_amt > stock_instance['Close']:
		new_stock_buy = stock(stock_instance['Date'], stock_instance['Close'])
		Stock_Inventory.buy_stock(new_stock_buy)
		buy_cash_amt = buy_cash_amt - stock_instance['Close']

	# for inv_stock in Stock_Inventory.stocks:
	# 	if inv_stock.price < 0.8 * stock_instance['Close']:
	# 		Stock_Inventory.sell_stock(inv_stock.date, stock_instance['Close'])

	for j in range(0, len(Sell_Learners)):
		pred_y_sell = predict_y(Sell_Learners[j], stock_instance) # get a base learner's sell pred
		if pred_y_sell == 1: sell_pos_vote += Sell_Learners[j]['weight']
		else: sell_neg_vote += Sell_Learners[j]['weight']

	if sell_pos_vote > sell_neg_vote and len(Stock_Inventory.stocks) > 0: 
		print 'Selling {0} stocks at {1:.2f} on {2}'.format(
			len(Stock_Inventory.stocks), stock_instance['Close'], stock_instance['Date'])
		Stock_Inventory.sell_stocks(stock_instance['Close'])

print "Summary:"
Stock_Inventory.sell_stocks(Test_Set[-1]['Close'])
print "Total value at beginning = {0}".format(INIT_CASH)
print "Total value at end = {0}".format(Stock_Inventory.cash)
periods = len(Test_Set)/365.0
print "value accrued at end = {0} over {1} years".format(Stock_Inventory.cash - INIT_CASH, periods)
print "Total Growth Rate: {0:.2f}%".format(100.0 * ((Stock_Inventory.cash - INIT_CASH)/INIT_CASH))
print "Average Annual Growth Rate: {0:.2f}%".format(100 * (Stock_Inventory.cash/INIT_CASH)/periods)

	
# print "Successfully classified {0} out of {1} instances ({2:.2f}%).".format(
# 			TP_count + TN_count, len(T), (TP_count+TN_count)*100.0/len(T))
# print "True Positive Rate: {0} out of {1} opportunities identified. ({2:.2f}%).".format(
# 			TP_count, TP_count + FN_count, TP_count*100.0/(TP_count + FN_count))

# print "Confusion Matrix:"
# print "  Actual:       Predicted:      "
# print "               Pos:      Neg:   "
# print "     Pos:      {0}        {1}".format(TP_count, FN_count)
# print "     Neg:      {0}        {1}\n".format(FP_count, TN_count)