import pickle
import setting
import numpy as np
import random
import os
## process data
## directory is 'processed_features/'
## name of each file 'paper','author',...
## format : data [tab] data [tab] ... [new_line]

DATA_TYPE = 'DBLP'
DATA_SIZE = 10728
HEADER = list('abcdefghijklmnlfjelkjflkejsl')

def np_array_to_text(array):
	txt = ''
	if isinstance(array,np.ndarray): array = array.tolist()
	for lst in array:
		txt += '\t'.join([str(item) for item in lst])+'\n'
	return txt

def make_dummy():
	data_type = setting.DATA_TO_USE[setting.DATA_SOURCE].split('_')

	data_val =  [[np.random.randn(random.choice([5,8,10,13])) for _ in range(DATA_SIZE)]] + [np.random.randn(DATA_SIZE,5)]*3 + [np.random.randn(DATA_SIZE,10)]

	for k,v in zip(data_type, data_val):
		with open('data/DBLP/processed_features/{}'.format(k),'w') as f:
			f.write('\t'.join(HEADER[:v[0].shape[0]])+'\n'+np_array_to_text(v))

def get_author_features():
	author_hindex = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorHindex.pickle'),'rb'))
	author_hindex_delta = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorHindexDelta.pickle'),'rb'))
	author_num_publication = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorPapers.pickle'),'rb'))
	author_mean_citation = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorMeanCitations.pickle'),'rb'))
	author_mean_citation_delta = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorMeanCitationDelta.pickle'),'rb'))
	author2paperid = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorAndPaperId.pickle'),'rb'))

	author_feature = {k:[author_hindex[k],author_hindex_delta[k]['delta'],author_num_publication[k],author_mean_citation[k],author_mean_citation_delta[k]] for k in author_hindex.keys()}

	# set unknown author's publication number to zero
	unknown_author = [k for k in author2paperid.keys() if len(author2paperid[k])>2000][0]
	author_feature[unknown_author][2] = 0

	paper_level_feature = {}

	for k,v in author2paperid.items():
		# whole_paper_ids.update(v)
		for pid in v:
			if pid in paper_level_feature:
				paper_level_feature[pid].append(author_feature[k]) 
			else: 
				paper_level_feature[pid] = [author_feature[k]]

	# take max values among co-authors for each feature 
	for k in paper_level_feature.keys():
		paper_level_feature[k] = np.array(paper_level_feature[k],dtype=np.float32).max(0)

	paper_ids = [l.strip() for l in open(os.path.join(setting.DATA_DIR[DATA_TYPE],'id_larger_than_5_compatible_with_abstract_v2.txt')).readlines()]

	ordered_features = [paper_level_feature[int(pid)] for pid in paper_ids]
	ordered_features = ['\t'.join([str(i) for i in l]) + '\n' for l in ordered_features]
	return ''.join(ordered_features)

def get_venue_features():
	venue_hindex = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venueHindex.pickle'),'rb'))
	venue_hindex_delta = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venueHindexDelta.pickle'),'rb'))
	venue_num_publication = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venuePapers.pickle'),'rb'))
	venue_mean_citation = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venueMeanCitation.pickle'),'rb'))
	venue_mean_citation_delta = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venueMeanCitationDelta.pickle'),'rb'))
	venue2paperid = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venue2PaperId.pickle'),'rb'))

	venue_feature = {k:[venue_hindex[k],venue_hindex_delta[k],venue_num_publication[k],venue_mean_citation[k],venue_mean_citation_delta[k]] for k in venue_hindex.keys()}

	# set empty venue's hindex and hindexDelta to zero
	venue_feature[''] = [0 for _ in range(5)]

	paper_level_feature = {}

	for k,v in venue2paperid.items():
		for pid in v:
			if pid in paper_level_feature:
				paper_level_feature[pid].append(venue_feature[k]) 
			else: 
				paper_level_feature[pid] = [venue_feature[k]]

	# take max values among co-authors for each feature 
	for k in paper_level_feature.keys():
		paper_level_feature[k] = np.array(paper_level_feature[k],dtype=np.float32).max(0)

	paper_ids = [l.strip() for l in open(os.path.join(setting.DATA_DIR[DATA_TYPE],'id_larger_than_5_compatible_with_abstract_v2.txt')).readlines()]
	ordered_features = [paper_level_feature[pid] for pid in paper_ids]
	ordered_features = ['\t'.join([str(i) for i in l]) + '\n' for l in ordered_features]
	return ''.join(ordered_features)

def get_paper_features():
	with open(os.path.join(setting.DATA_DIR[DATA_TYPE],'history_larger_than_5_compatible_with_abstract_v2.txt')) as f: history = f.readlines()

	def get_mean_citation(line):
		lsplit = line.split('\t')
		return float(lsplit[-1].strip())/len(lsplit)

	def get_diff(line,pos1,pos2):
		lsplit = [int(l.strip()) for l in line.split('\t')]
		if (pos2 == -2 and len(lsplit) > 1) or (pos2 == -3 and len(lsplit) > 2):
			return float(lsplit[pos1]-lsplit[pos2])
		else:
			try: return float(lsplit[pos1])
			except: return 0.0

	# paper_age, paper_mean_citation_per_year, paper_delta_1999, paper_delta_1998, paper_cumulative_citation
	paper_age = [11.0 + len(l.split('\t')) for l in history]
	paper_mean_citation_per_year = [get_mean_citation(l) for l in history]
	paper_delta_0 = [get_diff(l,-1,-2) for l in history]
	paper_delta_1 = [get_diff(l,-2,-3) for l in history]
	paper_cumulative_citation = [float(l.split('\t')[-1].strip()) for l in history]

	ordered_features = ['\t'.join([str(j) for j in i])+'\n' for i in zip(paper_age,paper_mean_citation_per_year,paper_delta_0,paper_delta_1,paper_cumulative_citation)]
	return ''.join(ordered_features)

def make_dblp_features():
	history = 'HEADER\n' + open('data/DBLP/history_larger_than_5_compatible_with_abstract_v2.txt').read()
	response = open('data/DBLP/response_larger_than_5_compatible_with_abstract_v2.txt').read()
	author = 'HEADER\n'+get_author_features()
	venue = 'HEADER\n'+get_venue_features()
	paper = 'HEADER\n'+get_paper_features()

	data_dict = {'history':history,
				'response':response,
				'author': author,
				'venue': venue,
				'paper': paper}

	for k,v in data_dict.items():
		with open(os.path.join(setting.DATA_DIR[DATA_TYPE],'processed_features',k), 'w') as f:
			f.write(v)

def make_kdd_features():
	pass

make_dblp_features()
