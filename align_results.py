import os
import re

import numpy as np

DIR = os.path.join(os.getcwd(),'results/kdd')
HEADER = ['main_module','text_encoder','category_embedding','attention']
EXTRA_HEADER = ['mape_%d'%(i) for i in range(1,11)] + ['r2_%d'%(i) for i in range(1,11)]
REGEX = re.compile(r'\d{1,2}[.]txt$')

def parse_result(fname):

	with open(os.path.join(DIR,fname)) as f: 
		mape,r2,args = [line.split(':')[1].strip() for line in f.readlines()]
	mape,r2 = eval(mape),eval(r2)
	mape+=['-1' for _ in range(10-len(mape))]
	r2+=['-1' for _ in range(10-len(r2))]

	ret = {}
	fsplit = fname.split('_')
	ret['repeat'] = fsplit[-1].split('.')[0]
	if fname.startswith('2rnn'):
		ret['aux'] = fsplit[0]
		values = fsplit[1:-1] + ['-' for _ in range(6-len(fsplit))]
		ret.update(dict(zip(HEADER + EXTRA_HEADER, values + [str(i) for i in mape + r2])))
	else:
		ret['aux'] = '-'
		values = fsplit[:-1] + ['-' for _ in range(5-len(fsplit))]
		ret.update(dict(zip(HEADER + EXTRA_HEADER, values + [str(i) for i in mape + r2])))

	return ret 

def to_tsv(dict_results):
	tsv = ''

	merged_results = {}
	for result in dict_results:
		line = [result[h] for h in HEADER+['aux']+EXTRA_HEADER]

		if str(line[:5]) not in merged_results:
			merged_results.update({str(line[:5]):[line[5:]]})
		else:
			merged_results[str(line[:5])].append(line[5:])

	# average results
	for k,v in merged_results.iteritems():
		# print v
		v = np.array([[float(val) for val in v_] for v_ in v])
		avg = v.mean(axis=0).tolist()
		std = v.std(axis=0).tolist()
		line = eval(k)+['%.3f(%.3f)'%(avg[i],std[i]) if avg[i] != -1 else '-' for i in range(len(avg)) ]
		tsv += '\t'.join(line)+'\n'

	return tsv


def write_tsv(fname,tsv):
	with open(fname,'w') as f: f.write(tsv)

results = os.listdir(DIR)
dict_results = []
for item in results:
	if len(REGEX.findall(item)) != 0 :
		dict_results.append(parse_result(item))

result_tsv = to_tsv(dict_results)
write_tsv('aligned_result.txt',result_tsv)

