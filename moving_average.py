# Will Dolan
# Spring 2017
# -*- coding: utf-8 -*-

## Moving data aggregate infos
## takes csv of stock info, outputs csv of following:
## Symbol, Date, ShortMA/Close, LongMA/Close, ShortMA/LongMA, 
##    Daily Change in Short MA, 3 Day Change in Short MA, 
##    Daily Change in Price, 3 Day Change in Price, 
##    3 Day Price Variance, 3 Day ShortMA Variance, CLASS
##    
## changes and variance are calculated 3 days prior
## CLASS is calculated to be whether there would have been
## an opportunity for > 15% growth on the stock within a month.
## It is given as -1 or 1.
## takes flag -buy or -sell to determine whether class is 'good buy'
## or 'good sell'

import sys
import random
import math
import numpy

# these represent periods
shortMA = 15
longMA = 50

output_keys = ['Symbol', 'Date', 'ShortMA', 'LongMA', 'Close',\
  'ShortMA/Close', 'LongMA/Close', 'ShortMA/LongMA', \
  'Daily Change in Short MA', '3 Day Change in Short MA', \
  'Daily Change in Price', '3 Day Change in Price', '3 Day Price Variance',\
  '3 Day ShortMA Variance', 'Spread',\
  'Dividend Yield',	'Price/Earnings', 'Earnings/Share', 'Book Value', 'Market Cap',\
  'EBITDA', 'Price/Sales', 'Price/Book', 'CLASS']

def process(filename, ref_data, buy_sell):
	#### File input, get keys
	print('reading file: ' + filename)
	f = open('./historical-values/' + filename, 'r')
	data = f.readlines()
	keys = data[0].split(',')
	if filename.endswith('.csv'):
	    symbol = filename[:-4]
	f.close()

	### Put input into key value pairs
	cleaned_data = []
	for line in data[1:]:
		vals = line.split(',')
		try:
			vals[2:] = map(lambda x: float(x), vals[2:])
			new_item = {}
			for i in range(0,len(keys)):
				new_item[keys[i]] = vals[i]
			cleaned_data.append(new_item)
		except ValueError:
			pass

	### initialize output array
	output = []
	symbol_ref_data = ref_data[symbol]
	for input_item in cleaned_data[longMA:]:
		init_output = {'Symbol':symbol, 'Date':input_item['Date'], \
						'ShortMA':0, 'LongMA':0, 'Close': input_item['Close'],\
						'ShortMA/Close':0, 'LongMA/Close':0,\
						'ShortMA/LongMA':0, 'Daily Change in Short MA':0, \
						'3 Day Change in Short MA':0, 'Daily Change in Price':0, \
						'3 Day Change in Price':0, '3 Day Price Variance':0, \
						'3 Day ShortMA Variance':0, 'Spread':0, \
						'Dividend Yield':symbol_ref_data['Dividend Yield'], 'Price/Earnings':symbol_ref_data['Price/Earnings'],\
						'Earnings/Share':symbol_ref_data['Earnings/Share'], 'Book Value':symbol_ref_data['Book Value'],\
						'Market Cap':symbol_ref_data['Market Cap'], 'EBITDA':symbol_ref_data['EBITDA'], \
						'Price/Sales':symbol_ref_data['Price/Sales'], 'Price/Book':symbol_ref_data['Price/Book'],\
						'CLASS':-1}
		output.append(init_output)

	## Now we calculate the Moving Averages
	## beware of indexing here its very confusing bc of adj for 50 day period but best i got
	i = longMA
	while i < len(cleaned_data):
		longMA_subset = cleaned_data[i-longMA:i]
		longMA_subset_closevals = map(lambda x: x['Close'], longMA_subset)
		shortMA_subset = cleaned_data[i-shortMA:i]
		shortMA_subset_closevals = map(lambda x: x['Close'], shortMA_subset)
		
		longMA_val = sum(longMA_subset_closevals)/longMA
		shortMA_val = sum(shortMA_subset_closevals)/shortMA
		output[i-longMA]['LongMA'] = longMA_val
		output[i-longMA]['ShortMA'] = shortMA_val
		output[i-longMA]['Close'] = cleaned_data[i]['Close']
		output[i-longMA]['Spread'] = (cleaned_data[i]['High'] - cleaned_data[i]['Low'])/ \
										cleaned_data[i]['Close']
		i += 1

	## Now for the fun arithmetic for all the attributes
	for i in range(4, len(output[3:]) - 1):
		output[i]['ShortMA/Close'] = output[i]['ShortMA']/output[i]['Close']
		output[i]['LongMA/Close'] = output[i]['LongMA']/output[i]['Close']
		output[i]['ShortMA/LongMA'] = output[i]['ShortMA']/output[i]['LongMA']
		output[i]['Daily Change in Short MA'] = output[i]['ShortMA']/output[i-1]['ShortMA']
		output[i]['3 Day Change in Short MA'] = output[i]['ShortMA']/output[i-3]['ShortMA'] 
		output[i]['Daily Change in Price'] = output[i]['Close']/output[i-1]['Close']
		output[i]['3 Day Change in Price'] = output[i]['ShortMA']/output[i-3]['ShortMA']
		output[i]['3 Day Price Variance'] = numpy.var(map(lambda x: x['Close'], output[i-3:i]))
		output[i]['3 Day ShortMA Variance'] = numpy.var(map(lambda x: x['ShortMA'], output[i-3:i]))

	## calculate class- whether there was a chance for 15% gain in 1mo
	## this behavior is contingent on whether buy_sell is "buy" or "sell"
	## determined by whether there is an item within a month that has 
	## 1.15x the current price (should buy now), or whether this is the highest price
	## in the upcoming month.
	for i in range(0, len(output[3:-30]) - 1):
		found = False
		for item in output[i+3:i+30]:
			if buy_sell == "-buy":
				change = item['Close']/output[i+3]['Close']
				if change > 1.15: 
					found = True
					break
			elif buy_sell == "-sell":
				if item['Close'] > output[i+3]['Close']:
					found = True
					break
		if buy_sell == "-buy" and found is True: output[i+3]['CLASS'] = 1
		elif buy_sell == "-sell" and found is False: output[i+3]['CLASS'] = 1
		
	## last let's normalize the close and MA figures with avg over whole period
	avgShortMA = sum(map(lambda x: x['ShortMA'], output))/len(output)
	avgLongMA = sum(map(lambda x: x['LongMA'], output))/len(output)
	avgClose = sum(map(lambda x: x['Close'], output))/len(output)
	for item in output:
		item['ShortMA'] /= avgShortMA
		item['LongMA'] /= avgLongMA
		item['Close'] /= avgClose

	## shave off first three days and last month
	output = output[4:-30]

	## File output
	# for item in output:
	# 	buff = ''
	# 	for key in output_keys:
	# 		buff += str(item[key]) + ','

	## option to write to a file
	f = open(('./results/' + symbol + '_output.csv'), 'w')
	# buff = ''
	# for key in output_keys:
	# 	buff += key + ','
	# f.write(buff[:-1]+'\n')

	for item in output:
		buff = ''
		for key in output_keys:
			buff += str(item[key]) + ','
		f.write(buff[:-1]+'\n')
	f.close

