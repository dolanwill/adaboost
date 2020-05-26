# Will Dolan
# Spring 2017
# AdaBoost
# -*- coding: utf-8 -*-

# Readme:
# This is an adapted implementation of the AdaBoost.M1 meta-algorithm.
# It takes two .csv files as input:
# a training set with a set of attributes x1 to xn-1, and its classification as xn for n columns.
# a test set with the same set of attributes x1 to xn for n columns.
# 
# usage:
# python adaboost.py [training.csv] [test.csv] [optional: L specification ex. 8 (number of stump implementation learners to output.)]
# if L is not specified as an argument, the user will be prompted at runtime. 

# In testing, I have found that values of L between 6-10 work best and
# consistently classify between 65-75% of test cases correctly, but this is
# specific to the dataset.

# In this use case, I have found that values of L between 6-10 work best and
# consistently classify between 73-75% of test cases correctly, but this is
# specific to the dataset

import sys
import random
import math

### Summary for building the base learners: implemented as Decision Stumps
### 		Algorithm: On a dataset X of form [ [x1, x2, .. x_n, y], ... ]
###   		iterate through decider Decision Stump (DS) on each attribute x to determine 
###         what has best accuracy.
###  		Algorithm for DS: set initial threshold = avg value.
###            Recursively optimize threshold classification
###				accuracy = (true pos + true neg)/total until no increase in accuracy
###			Threshold adjusted by incr/decr it by +/- 1%.

# global var for how many attrs to not convert to float - put first in csv.
# (data specific)
attr_buffer = 2
# todo this does not belong here, make a straight adaboost implementation and 
# force consumer to adhere to api
keys = []


# returns dict of form 
# {'attr': float, 'attr_name': str, 'acc': float, 'threshold': float, 'dir': str}

def buildDS(X):
	best_attr = {'attr': 0, 'acc': 0, 'threshold': 0, 'dir': '', 'attr_name':''}
	for i in range(attr_buffer, attr_count):
		attr_class_zip = []
		for instance in X:
			attr_class_zip.append([instance[i], instance[attr_count]])

		optimal_classifier = getOptimalThreshold(attr_class_zip)

		if optimal_classifier['acc'] > best_attr['acc']:
			best_attr = optimal_classifier
			best_attr['attr'] = i
			best_attr['attr_name'] = keys[i]
	return best_attr

### This function takes a dataset of form [[x,y], [x,y]...] and returns the
### best threshold for predicting y. It also returns the direction the
### positive classifications are from the threshold and the accuracy.
def getOptimalThreshold(X):
	(attrs, classes) = zip(*X)
	Neg = filter(lambda (attr, label): label < 0, X)
	Pos = filter(lambda (attr, label): label > 0, X)
	neg_ct = len(Neg)
	pos_ct = len(Pos)

	initial_threshold = sum(attrs)/len(attrs) 
	# initializes to average value of attributes
	# alternatively - can initialize to random value
	# with threshold = random.choice(attrs) - no change in performance.

	###	Algorithm for DS: set initial threshold = avg value.
	###         Recursively optimize threshold classification
	###		   accuracy = (true pos.+ true neg)/total by moving threshold
	###		   by 2% increase/decrease step.
	###		   We get true pos. true neg. rates by filtering by label and counting.
	def rectestThreshold(X, threshold, baselineAccuracy):
		# partition by position rel. to threshold, then by true label.
		labeled_lt = filter(lambda (attr, label): attr < threshold, X)
		labeled_gt = filter(lambda (attr, label): attr >= threshold, X)
		pos_labeled_lt = filter(lambda (attr, label): label > 0, labeled_lt)
		neg_labeled_lt = filter(lambda (attr, label): label < 0, labeled_lt)
		pos_labeled_gt = filter(lambda (attr, label): label > 0, labeled_gt)
		neg_labeled_gt = filter(lambda (attr, label): label < 0, labeled_gt)

		# determine which side of the threshold most positive labels are
		if float(len(pos_labeled_lt))/pos_ct > 0.5:
			pos_direction = 'lt'
		else:
			pos_direction = 'gt'

		# determine accuracy by seeing how many were labeled correctly.
		if pos_direction == 'lt':
			TP_rate = len(pos_labeled_lt)
			TN_rate = len(neg_labeled_gt)
		else:
			TP_rate = len(pos_labeled_gt)
			TN_rate = len(neg_labeled_lt)
		newAccuracy = float(TN_rate + TP_rate)/len(X)

		# if we are improving accuracy, continue to optimize by recursively calling
		# if not, return current accuracy
		if newAccuracy > baselineAccuracy:
			incr = rectestThreshold(X, threshold*1.01, newAccuracy)['acc']
			decr = rectestThreshold(X, threshold*0.99, newAccuracy)['acc']
			if incr > decr:
				return {'acc':incr, 'threshold':threshold*1.01,\
						 'dir':pos_direction}
			else:
				return {'acc':decr, 'threshold':threshold*0.99,\
						 'dir':pos_direction}
		else:
			return {'acc':baselineAccuracy, 'threshold':threshold,\
						 'dir':pos_direction}

	# Back in getOptimalThreshold: return key/value pair describing threshold.
	return rectestThreshold(X, initial_threshold, 0)

#### drawX: takes the set of Xs and their respective Probabilities P.
###    Creates a set of instances X_j that are drawn from X according to
###    when a randomly generated float between [0,1) corresponds to the
###    cumulative distribution up to the instance x's position. Points with
###    a higher corresponding probability will have a higher chance of being
###    picked. Returns the multiset X_j
def drawX(X, P):
	X_j = []
	for i in range(0, len(X) - 1):
		randval = random.random()
		p_sum = 0
		j = 0
		while randval > p_sum:
			try:
				p_sum += P[j]
				j += 1
			except IndexError:
				j = len(X) - 1
				break
		X_j.append(X[j-1])
	return X_j

### Given a decider with threshold and pos direction and an instance, 
### predict the instance's label using its attribute.
def predict_y(decider, instance):
	instance_attr = instance[decider['attr']]

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

### File sanitize: convert to floats, return dataset
def clean_data(data, keys):
	cleaned_data = []
	for line in data[1:]:
		vals = line.split(',')

		# attr_buffer, globally specified, is how many attr at start to ignore
		vals[attr_buffer:] = map(lambda x: float(x), vals[attr_buffer:])
		new_item = {}
		for i in range(0,len(vals)):
			new_item[i] = vals[i]
		cleaned_data.append(new_item)
	return cleaned_data

######################################################################
######### Main/Upper Level functionality:
#### Import files from command line arg
f1 = open(sys.argv[1])
training_input = f1.readlines()
f1.close()
keys = training_input[0].split(',')
attr_count = len(keys) - 1
X = clean_data(training_input, keys)

### Now, for each base learner specified by interactive input or third arg,
### run the AdaBoost algorithm.

try: L = int(sys.argv[3])
except(IndexError): 
	L = int(input("Please input the desired number of base-learners, L: "))
print "Training {0} learners...\n".format(L)

### Training:
### For all instances (x,r), initialize probabilities to 1/N
P = []
for i in X:
	P.append(1.0/len(X))

BaseLearners = []
Accuracies = []

l = 0
while l < L:
	X_j = drawX(X, P)								# Draw multiset X_j
	optimizedDS = buildDS(X_j)						# Build dec. stump decider

	pred_labels = []
	error_rate = 0

	for i in range(0, len(X) - 1):					# Now find predicted labels
		pred_labels.append(predict_y(optimizedDS, X[i]))
		real_label = X[i][attr_count]

		if pred_labels[i] != real_label:			# if pred != real
			error_rate += P[i]						# increase error rate

	Beta_j = error_rate / (1.0 - error_rate)		# Get decider's accuracy
	for i in range(0, len(X) - 1):					# Now update probabilities
		real_label = X[i][attr_count]
		if pred_labels[i] == real_label:			# if pred == real
			P[i] *= Beta_j							# reduce prob. for next rnd

	Z_j = sum(P)									# renormalize probabilities
	for i in range(0, len(P) - 1):
		P[i] = P[i] / Z_j

	print 'Learner {0}: '.format(l+1),\
	    'Classifying on {0}:\n '.format(optimizedDS['attr_name']),\
	    '    Weight {0:.3f}, '.format(-1.0 * math.log(Beta_j)),\
	    'Threshold {0:.3f}, '.format(optimizedDS['threshold']),\
	    'error rate {0:.3f}\n'.format(error_rate)

	if error_rate > 0.5:							# Ignore if too inaccurate
		print "         Error rate exceeds 0.5. Reassigning Weights and redoing."
		pass
	else:
		BaseLearners.append(optimizedDS)			# If accurate enough, save.
		Accuracies.append(Beta_j)
		l += 1


##############################################################################
### Applying to Test Set
try: testfile = sys.argv[2]
except(IndexError): sys.exit(1)

print "\nApplying trained learners to test data. \n"
f2 = open(testfile)
test_input = f2.readlines()
f2.close()

T = clean_data(test_input, keys)

TP_count = TN_count = FP_count = FN_count = 0
for instance in T:								# predict label of each x
	neg_vote = pos_vote = 0
	for j in range(0, L):
		pred_y_j = predict_y(BaseLearners[j], instance) # get each BL's pred
		Beta_j = Accuracies[j]
														# add vote to total
														# (using log vote w/ weight)
		if pred_y_j == 1: pos_vote += math.log(1.0/Beta_j) 
		else: neg_vote += math.log(1.0/Beta_j)

	if pos_vote > (0.5) * neg_vote: pred_y = 1   		# now count votes
														# 0.5 for generosity
	else: pred_y = -1									# to get output pred

	if pred_y == instance[attr_count]:
		if pred_y == 1:
			TP_count += 1
		else: TN_count += 1
	elif pred_y == 1: FP_count += 1
	else: FN_count += 1

	
print "Successfully classified {0} out of {1} instances ({2:.2f}%).".format(
			TP_count + TN_count, len(T), (TP_count+TN_count)*100.0/len(T))
print "True Positive Rate: {0} out of {1} opportunities identified. ({2:.2f}%).".format(
			TP_count, TP_count + FN_count, TP_count*100.0/(TP_count + FN_count))

print "Confusion Matrix:"
print "  Actual:       Predicted:      "
print "               Pos:      Neg:   "
print "     Pos:      {0}        {1}".format(TP_count, FN_count)
print "     Neg:      {0}        {1}\n".format(FP_count, TN_count)


