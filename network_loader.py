#!/usr/bin/env python  

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import img2pdf

def loadGame_f2f(network=62, sec=5, src='./comm-f2f-Resistance' ):
	
	adjacency = np.zeros((451, 451))
	partcipants = pd.read_csv(f'{src}/network_list.csv')
	start = 0

	time_span = list(np.arange(0,sec*3))
	for idx, game in enumerate(np.arange(network)):
		df_network = pd.read_csv(f'{src}/network/network{game}.csv')
		partcipant = partcipants[ partcipants['NETWORK']==idx ]['NUMBER_OF_PARTICIPANTS'].values[0]
		tri_group = np.zeros((partcipant,partcipant))
		group = np.zeros(partcipant*partcipant)
		count = 0
		for column in df_network.columns: 
			if (column[-3:] != 'TOP') and (column != 'TIME'): 
				contact = 1 if df_network[ df_network['TIME'].isin(time_span) ][column].values.sum() > 0 else 0
				group[count] = contact
				count += 1
		group = group.reshape(partcipant,partcipant)
		for i in range(partcipant):
			for j in range(partcipant):
				if group[i,j] == 1:
					tri_group[min(i,j),max(i,j)] = 1
		adjacency[start:start+partcipant, start:start+partcipant] = tri_group
		start += partcipant

	if sec < 1:
		sec_info = str(sec)[0] + '_' + str(sec)[-1]
	else:
		sec_info = secs
	np.save(f'game/period_{sec_info}_secs.npy', adjacency)
	return adjacency



def loadGame_fb(limit=500, seed=0):
	adjacency = np.zeros((limit, limit))
	np.random.seed(seed)
	candidates = sorted(np.random.randint(0, 4039, size=limit))
	can_dic = {}
	r_can_dic = {}
	for cs, ids in enumerate(candidates):
		can_dic[ids] = cs
		r_can_dic[cs] = ids
		
	with open('./facebook_combined.txt') as f:
		lines = f.readlines()
		for line in lines:
			n1 = int(line.strip().split(' ')[0])
			n2 = int(line.strip().split(' ')[1])
			if (n1 in candidates) and (n2 in candidates):
				adjacency[can_dic[n1]][can_dic[n2]] = 1
	np.save(f'fb/network_{limit}_seed_{seed}.npy', adjacency)
	return adjacency, can_dic, r_can_dic






if __name__ == '__main__':

	limit, seed = 500, 0
	if not os.path.exists(f'fb/network_{limit}_seed_{seed}.npy'):
		adjacency, dics, rdics = loadGame_fb(limit,seed)
	else:
		adjacency = np.load(f'fb/network_{limit}_seed_{seed}.npy')
	adj = adjacency + adjacency.T




	# whos is social king
	for i in range(adj.shape[0]):
		print(i, rdics[i], int(np.sum(adj[i])))


	# degree distribution
	d = defaultdict(int)		
	for i in range(adj.shape[0]):
		degree = int(np.sum(adj[i]))
		d[degree] += 1
	edges = 0
	for key in sorted(d):
		edges += key*d[key]

	plt.figure(figsize=(8,6.5))   
	x = [key for key in sorted(d)[:-1]]
	plt.bar(x, height=[d[key] for key in sorted(d)[:-1]])
	plt.xticks(x, fontsize=12)
	plt.ylabel('count', fontsize=18)
	plt.yticks(fontsize=18)
	plt.xlabel('degree', fontsize=18)
	plt.title(f'total #edges = {edges}', fontsize=18)
	img_name = f'fb/network_{limit}_seed{seed}.jpg'
	plt.savefig(img_name)
	
	pdf_name = img_name[:-3] + 'pdf'
	pdf_bytes = img2pdf.convert(img_name)
	file_ = open(pdf_name, "wb")
	file_.write(pdf_bytes)
	file_.close()


	# plot the adj
	plt.figure(figsize=(8,8))   
	adjacency_norm = (adj-adj.min()) / (adj.max()-adj.min())
	for i in range(500):
		for j in range(500):
			if adjacency_norm[i][j] == 1:
				adjacency_norm[i][j] == 3
	
	inf = np.load('fb/samples/seed_0_ROUND_3_beta_1_ita_1.5_k_28.npy')
	for idx,val in enumerate(inf):
		if val == 1:
			adjacency_norm[idx,:] = 0.05
			adjacency_norm[:,idx] = 0.05
	
	plt.imshow(adjacency_norm, cmap='Set3')
	plt.savefig(f'fb/adjacency1.jpg')
	plt.close()














