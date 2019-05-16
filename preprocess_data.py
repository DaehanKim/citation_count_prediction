import pickle
import setting
import numpy as np
import random
import os
################################################
## codes are for processing data
## resulting directory is 'processed_features/'
## name of each file 'paper','author',...
## format : data [tab] data [tab] ... [new_line]
#################################################

DATA_TYPE = setting.DATA_SOURCE
TARGET_DIR = 'processed_features'

def np_array_to_text(array):
	txt = ''
	if isinstance(array,np.ndarray): array = array.tolist()
	for lst in array:
		txt += '\t'.join([str(item) for item in lst])+'\n'
	return txt

def get_author_features():

	author_hindex = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorHindex.pickle'),'rb'))
	author_hindex_delta = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorHindexDelta.pickle'),'rb'))
	author_num_publication = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorPapers.pickle'),'rb'))
	author_mean_citation = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorMeanCitations.pickle'),'rb'))
	author_mean_citation_delta = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorMeanCitationDelta.pickle'),'rb'))
	author2paperid = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','authorAndPaperId.pickle'),'rb'))

	# print(len(author_hindex))
	author_feature = {k:[author_hindex[k],author_hindex_delta[k]['delta'],author_num_publication[k],author_mean_citation[k],author_mean_citation_delta[k]] for k in author_hindex.keys()}
	# author_feature = {k:[author_pagerank_unweighted[k],author_pagerank_weighted[k]] for k in author_hindex.keys()}

	# set unknown author's publication number to zero
	try:
		unknown_author = [k for k in author2paperid.keys() if len(author2paperid[k])>2000][0]
		author_feature[unknown_author][2] = 0
	except:
		pass

	paper_level_feature = {}

	for k,v in author2paperid.items():
		# whole_paper_ids.update(v)
		for pid in v:
			if pid in paper_level_feature:
				paper_level_feature[str(pid)].append(author_feature[k]) 
			else: 
				paper_level_feature[str(pid)] = [author_feature[k]]

	# take max values among co-authors for each feature 
	for k in paper_level_feature.keys():
		paper_level_feature[k] = np.array(paper_level_feature[k],dtype=np.float32).max(0)

	# paper_ids = [l.strip() for l in open(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.ID_PATH[DATA_TYPE])).readlines()]
	paper_ids = [l.strip() for l in open(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.ID_PATH[DATA_TYPE])).readlines()]

	# print paper_ids[:10]
	# print paper_level_feature.keys()[:10]

	ordered_features = [paper_level_feature[pid] for pid in paper_ids]
	ordered_features = ['\t'.join([str(i) for i in l]) + '\n' for l in ordered_features]
	return ''.join(ordered_features)

def get_venue_features():
	venue_hindex = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venueHindex.pickle'),'rb'))
	venue_hindex_delta = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venueHindexDelta.pickle'),'rb'))
	venue_num_publication = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venuePapers.pickle'),'rb'))
	venue_mean_citation = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venueMeanCitation.pickle'),'rb'))
	venue_mean_citation_delta = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venueMeanCitationDelta.pickle'),'rb'))
	venue2paperid = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','venue2PaperId.pickle'),'rb'))

	# venue_feature = {k:[venue_hindex[k],venue_hindex_delta[k],venue_num_publication[k],venue_mean_citation[k],venue_mean_citation_delta[k]] for k in venue_hindex.keys()}
	venue_feature = {k:[venue_mean_citation[k],venue_mean_citation_delta[k]] for k in venue_hindex.keys()}

	# set empty venue's hindex and hindexDelta to zero
	venue_feature[''] = [0 for _ in range(2)]

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

	# paper_ids = [l.strip() for l in open(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.ID_PATH[DATA_TYPE])).readlines()]
	paper_ids = [l.strip() for l in open(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.ID_PATH[DATA_TYPE])).readlines()]
	
	ordered_features = [paper_level_feature[pid] for pid in paper_ids]
	ordered_features = ['\t'.join([str(i) for i in l]) + '\n' for l in ordered_features]
	return ''.join(ordered_features)

def get_paper_features():
	history_path = os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.HISTORY_PATH[DATA_TYPE])
	with open(history_path) as f: history = f.readlines()
	# innovativeness_path = os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'innovativeness.txt')
	# with open(innovativeness_path) as f: innovativeness = f.readlines()

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
	# innovativeness = [float(i) for i in innovativeness]

	author_pagerank_weighted = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','pageRankWithWeight.pickle'),'rb'))
	author_pagerank_unweighted = pickle.load(open(os.path.join(setting.DATA_DIR[DATA_TYPE],'features','pageRankWithoutWeight.pickle'),'rb'))

	paper_ids = [l.strip() for l in open(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.ID_PATH[DATA_TYPE])).readlines()]

	pagerank_weighted = [author_pagerank_weighted[int(pid)] if int(pid) in author_pagerank_weighted else 0.0 for pid in paper_ids]
	pagerank_unweighted = [author_pagerank_unweighted[int(pid)] if int(pid) in author_pagerank_unweighted else 0.0 for pid in paper_ids]
	ordered_features = ['\t'.join([str(j) for j in i])+'\n' for i in zip(paper_age,paper_mean_citation_per_year,paper_delta_0,paper_delta_1,paper_cumulative_citation)]
	# ordered_features = ['\t'.join([str(j) for j in i])+'\n' for i in zip(innovativeness)]
	
	return ''.join(ordered_features)

def get_paper_embedding():
	# config
	embedding_path = os.path.join(setting.DATA_DIR[DATA_TYPE],'paper_embedding','word_embedding_{}.txt'.format(setting.EMBEDDING_DIM))
	# id_path = os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.ID_PATH[DATA_TYPE])
	id_path = os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,setting.ID_PATH[DATA_TYPE])

	with open(embedding_path) as f:
		embedding = f.readlines()[1:]
	embedding_dict = {line.split()[0]:'\t'.join(line.split()[1:])+'\n' for line in embedding}
	paper_ids = [l.strip() for l in open(id_path).readlines()]
	ordered_features = [embedding_dict[k] for k in paper_ids]
	return ''.join(ordered_features)

def extract_enough_cited_id_history_response():
	paper_ids = [l.strip() for l in open(os.path.join(setting.DATA_DIR[DATA_TYPE],setting.ID_PATH[DATA_TYPE])).readlines()]
	paper_history = open(os.path.join(setting.DATA_DIR[DATA_TYPE],setting.HISTORY_PATH[DATA_TYPE])).readlines()
	paper_response = open(os.path.join(setting.DATA_DIR[DATA_TYPE],setting.RESPONSE_PATH[DATA_TYPE])).readlines()

	enough_cited_paper_id = []
	enough_cited_paper_history = []
	enough_cited_paper_response = []
	for idx,(id, history_line) in enumerate(zip(paper_ids, paper_history)):
		last_history = int(history_line.split()[-1].strip())
		if last_history >= 5:
			enough_cited_paper_id.append(id)
			enough_cited_paper_history.append(paper_history[idx])
			enough_cited_paper_response.append(paper_response[idx])

	if not os.path.exists(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR)):
		os.mkdir(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR))

	result_file_path = {
						'ID':os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.ID_PATH[DATA_TYPE]),
						'HISTORY':os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.HISTORY_PATH[DATA_TYPE]),
						'RESPONSE':os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.RESPONSE_PATH[DATA_TYPE])
						}

	with open(result_file_path['ID'],'w') as f: 
		f.write('\n'.join(enough_cited_paper_id)+'\n')
	with open(result_file_path['HISTORY'],'w') as f: 
		f.write(''.join(enough_cited_paper_history))
	with open(result_file_path['RESPONSE'],'w') as f: 
		f.write(''.join(enough_cited_paper_response))


def make_features():
	history = open(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.HISTORY_PATH[DATA_TYPE])).read()
	response = open(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,'enough_cited_'+setting.RESPONSE_PATH[DATA_TYPE])).read()
	author = get_author_features()
	venue = get_venue_features()
	paper = get_paper_features()
	# paper_embedding = get_paper_embedding()

	data_dict = {'history':history,
				'response':response,
				# 'author-abstract': author,
				'venue': venue,
				'author': author,
				'paper': paper}

				# 'paper_embedding':paper_embedding}

	# data_dict = {'venue':venue}
	for k,v in data_dict.items():
		with open(os.path.join(setting.DATA_DIR[DATA_TYPE],TARGET_DIR,k), 'w') as f:
			f.write(v)

extract_enough_cited_id_history_response()
make_features()
