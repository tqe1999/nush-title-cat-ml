import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import msgpack
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

# Create a list of the most common words (lemmatized)
def create_lexicon(data):
	lexicon = []
	with open(data, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l.lower())
			lexicon += list(all_words)


	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	#w_counts = {'the':52212, 'and':25242}
	l2 = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50: # We do not want super common words or obscure words
			l2.append(w)
	print(len(l2))
	return l2

# Get the features of each product
def sample_handling(dataFile, lexicon, clarityFile, concisenessFile):
	featureset = [] #featureset is an array of [features, [clarity, conciseness]]
	clarity = []
	conciseness = []
	with open(clarityFile,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			clarity.append(int(l))

	with open(concisenessFile,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			conciseness.append(int(l))

	#with open(data_file, 'r') as csvfile:  # this will close the file automatically.
    #	reader = csv.reader(csvfile)
    #	for row in reader:
	#       		print row
	with open(dataFile,'r') as f:
		contents = f.readlines()
		i = 0 # For conciseness[i] and clarity[i]...ideally was hopping for for i in range(0,contents.count)
		for l in contents[:hm_lines]:
			print(i)
			# Tokenize words
			current_words = word_tokenize(contents[i].lower())
			# Lemmatize words
			current_words = [lemmatizer.lemmatize(j) for j in current_words]
			# Create array the size of the lexicon
			features = np.zeros(len(lexicon))
			# Add 1 to the index value in the features array that is the same index of the word in the lexicon
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			# Convert the array to a list / () -> []
			features = list(features)
			featureset.append([features, [clarity[i], conciseness[i]]])
			i += 1 # If anyone knows how to use an iterative in the for loop that will be great but I do not know how and I am pressed for time. - xy

	return featureset

# Function to call from testing
def create_feature_set_and_labels(dataFile, clarityFile, concisenessFile, test_size=0.1):
	lexicon = create_lexicon(dataFile)
	featureset = sample_handling(dataFile, lexicon, clarityFile, concisenessFile)
	random.shuffle(featureset)

	# Convert list back to array / [] -> ()
	featureset = np.array(featureset)

	testing_size = int(test_size*len(featureset)) # Amount of products set aside to test

	# Featureset: [[features,[clarity, conciseness]],[features,[clarity, conciseness]]]
	# All features and labels in respective lists for training (in this case, 90%)
	train_features = list(featureset[:,0][:-testing_size]) # [:,0] takes all the first elements (features), [:,1] takes the labels
	train_labels = list(featureset[:,1][:-testing_size]) # [:-testing_size] takes the first 90% products to train

	# All features and labels in respective lists for testing (in this case, 10%)
	test_features = list(featureset[:,0][-testing_size:])
	test_labels = list(featureset[:,1][-testing_size:])

	return train_features,train_labels,test_features,test_labels

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_set_and_labels('data_train.csv', 'clarity_train.labels', 'conciseness_train.labels')
	print('Saving msgpack')
	# Write msgpackfile
	with open('data.msgpack', 'wb') as outfile:
	#	print(len(train_x))
	#	print(len(train_y))
#		print(len(test_x))
#		print(len(test_y))
#		pickle.dump([train_x,train_y,test_x,test_y], handle, protocol=pickle.HIGHEST_PROTOCOL)
		msgpack.pack([train_x,train_y,test_x,test_y],outfile)
	print('Done!')














