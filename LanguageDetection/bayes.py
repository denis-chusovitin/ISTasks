# -*- coding: utf-8 -*-

import re
from math import log

n = 5 #n-grams
train_set = 0.9

lang_files = ['Russian.txt', 'Ukrainian.txt', 'Belarusian.txt', 'Bulgarian.txt',
'Macedonian.txt', 'Kazakh.txt']
lang_regexs = map(re.compile, [u'[^а-я]', u'[^абвгґдеєжзиіїйклмнопрстуфхцчшщьюя]',
u'[^аоуіэыяеёюбвгджзйклмнпрстўфхцчшь]', u'[^абвгдежзийклмнопрстуфхцчшщъьюя]',
u'[^абвгдѓежзиѕјклљмнњопрстќуфхцчџш	]', u'[^аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя]'])

lang_amount = len(lang_files)

data = [] * lang_amount
cond_probs = {}
lang_prob = [0.0] * lang_amount

additive_smoothing = 0.0000001

total_len = 0

def read_data():
	print "Reading data.."
	global total_len
	
	for i in range(lang_amount):
		print lang_files[i]
		
		data.append([])
		with open(lang_files[i], 'r') as f:
			for line in f:
				words = split_sentence(line, i)
				data[i].append(words)
				
				total_len += len(words)

	print "Done."

def split_sentence(sentence, language):
	words = sentence.split()
	
	return map(lambda w: \
		lang_regexs[language].sub('', unicode(w, 'utf-8').lower()), words)

def add_feature(f, i):
	if f not in cond_probs:
		cond_probs[f] = [0.0] * lang_amount
	cond_probs[f][i] += 1
	
def get_value(f, i):
	if f not in cond_probs:
		return 0
	else:
		return cond_probs[f][i]
	
def get_ngrams(sentence):
	res = []

	spaces = '_' * (n - 1)
		
	for word in sentence:
		word = spaces + word + spaces
		res += [word[i:i + n] for i in range(len(word) - n + 1)]
	return res
	

def train(words, language):
	ngrams = get_ngrams(words)
	lang_prob[language] += len(words)
	
	for feature in ngrams:
		add_feature(feature, language)
		
def normalize_data():
	for i in xrange(lang_amount):
		for feature in cond_probs:
			cond_probs[feature][i] /= lang_prob[i]
			
		lang_prob[i] /= total_len

def classify(words):
	features = get_ngrams(words)
	
	classifier = lambda l: sum([log(get_value(f, l) + additive_smoothing) for f in features])
	'''
	classifier = lambda l: log(lang_prob[l]) + \
		sum([log(get_value(f, l) + additive_smoothing) for f in features])
	'''
	res = map(classifier, xrange(lang_amount))
	return res.index(max(res))

read_data()

print "Training.."

for i in range(lang_amount):
	train_amount = int(train_set * len(data[i]))
	for j in xrange(train_amount):
		train(data[i][j], i)
					
normalize_data()

print "Done."

print "Testing.."

for i in range(lang_amount):
	k = 0
	train_amount = int(train_set * len(data[i]))
	for j in xrange(train_amount, len(data[i])):
		if classify(data[i][j]) == i:
			k = k + 1
	
	print 1.0 * k / (len(data[i]) - train_amount)
	
print "Done."

