from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import pandas as pd 
import sys
import random
from sklearn.metrics.cluster import normalized_mutual_info_score
import seaborn as sns
from sklearn.cluster import SpectralClustering
from scipy.spatial import distance
import numpy as np
import csv
from sklearn.decomposition import PCA


def getData(fname):
	df = pd.read_csv(fname, delimiter='\t')
	df = df.fillna(df.mean())

	# only the hon features
	f = range(0,6)

	df_classical = df[df['genre'] == 'CLASSICAL']
	df_jazz = df[df['genre'] == 'JAZZ']
	df_pop = df[df['genre'] == 'POP']
	df_rock = df[df['genre'] == 'ROCK']
	df_folk = df[df['genre'] == 'FOLK']

	df = pd.concat([df_classical, df_jazz, df_pop, df_rock, df_rock, df_folk])

	features = df.iloc[:,f].values
	labels = df.iloc[:,-1].values
	
	return features, labels


def getDataArtist(fname):
	df = pd.read_csv(fname, delimiter='\t')
	df = df.fillna(df.mean())

	# only the hon features
	f = range(0,6)

	df_0 = df[df['genre'] == 'MOZART']
	df_1 = df[df['genre'] == 'BACH']
	df_2 = df[df['genre'] == 'VIVALDI']
	df_3 = df[df['genre'] == 'BEATLES']
	df_4 = df[df['genre'] == 'NIRVANA']

	df = pd.concat([df_0, df_1, df_2, df_3, df_4])

	features = df.iloc[:,f].values
	labels = df.iloc[:,-1].values
	
	return features, labels


def principleComponents(features, labels):
	pca = PCA(n_components=3)
	x = pca.fit_transform(features)

	print('PCA Variance', pca.explained_variance_ratio_)

	return [np.append(x[i], [labels[i]]) for i in range(len(x))]


def saveData(sname, data):
	with open(sname, 'w+') as f:
		writer = csv.writer(f, delimiter='\t')
		for r in data:
			writer.writerow(r)


if __name__ == '__main__':
	fname = sys.argv[1]
	sname = sys.argv[2]
	mode = sys.argv[3]			# genre or artist

	if mode == 'genre':
		features, labels = getData(fname)
	elif mode == 'artist':
		features, labels = getDataArtist(fname)

	pc = principleComponents(features, labels)
	saveData(sname, pc)