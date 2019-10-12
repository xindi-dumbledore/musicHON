import csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import sys
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


def readFeatures(fname):
	data = {}
	with open(fname, 'r') as f:
		reader = csv.reader(f, delimiter='\t')
		next(reader)
		for row in reader:
			label = row[-1]

			#features = list(map(float, row[:-1])) 				# MusicHON+ and SimpleNetwork+
			#features = list(map(float, row[:-4]))				# SimpleNetwork
			features = row[0:7]									# MusicHON features
			features = list(map(float, features))
			
			features = [x if not np.isnan(x) else 0 for x in features] 		# reploce nan by 0 or mean

			if label not in data:
				data[label] = []

			data[label].append(features)
	return data

def sampling(data, labels=None):
	"""
	Down sample majority class to same size as minority
	"""
	X, y = [], []

	if labels is None:
		labels = set(data.keys())

	dcount = {l:len(data[l]) for l in labels}
	dsize = min(dcount.values())
	
	for l in labels:
		features = data[l]
		np.random.shuffle(features)
		features = features[:dsize]
		for d in features:
			X.append(d)
			y.append(l)

	return X, y


def classifyScoresBinary(clf, X, y):
	scores = cross_validate(clf, X, y, cv=5, scoring=['roc_auc'])

	return [np.mean(scores['test_roc_auc']), np.std(scores['test_roc_auc'])]


def saveClassificationResuts(sname, data):
	with open(sname, 'a+') as f:
		writer = csv.writer(f, delimiter='\t')
		for row in data:
			writer.writerow(row)


def featureImportance(X,y):
	clf = ExtraTreesClassifier(n_estimators=50)
	clf = clf.fit(X, y)
	print(clf.feature_importances_)


if __name__ == '__main__':
	fname = sys.argv[1]
	sname = sys.argv[2]
	cl  = sys.argv[3]			# classifier

	labelslist = [['CLASSICAL','FOLK'], ['CLASSICAL','JAZZ'],['CLASSICAL','POP'],['CLASSICAL','ROCK'],['FOLK','JAZZ'],['FOLK','POP'],['FOLK','ROCK'],['JAZZ','ROCK'],['JAZZ','POP'],['POP','ROCK']]
	#labelslist = [['MOZART','BACH'], ['MOZART','VIVALDI'],['MOZART','BEATLES'],['MOZART','NIRVANA'],['BACH','VIVALDI'],['BACH','BEATLES'],['BACH','NIRVANA'],['VIVALDI','BEATLES'],['VIVALDI','NIRVANA'],['BEATLES','NIRVANA']]

	data = readFeatures(fname)

	for labels in labelslist:
		X, y = sampling(data, labels)

		featureImportance(X,y)

		if cl == 'svm':
			clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
		elif cl == 'rf':
			clf = RandomForestClassifier(n_estimators=100)
		else:
			clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5, hidden_layer_sizes=(5, 5), learning_rate='adaptive')

		results = classifyScoresBinary(clf, X, y)
		
		print(results)
		