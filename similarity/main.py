from bag_of_word import BagOfWordSimilarity
from word_embed_sim import WordEmbeddedSimilarity
from service_helper import send_progress
import json 
import os
# This package is for downloading pdf
import urllib.request
from Util import Util
import sys
import ast

as_worker = True

if not as_worker:
    with open('similarity_json_request.json', 'r') as f:
        request_dict = json.load(f)
else:
    request = {
        'id': sys.argv[1],
        'project_id': sys.argv[2],
        'project_name': sys.argv[3],
        'documents': ast.literal_eval(sys.argv[4]),
        'criteria': int(sys.argv[5])
    }

# define a input local root
# input_local_root = '/Users/Kim/Documents/trf_dir/TestDownloadFiles/ori/'
input_local_root = '../documents/' + request['id'] + '/'
# converted_local_root = '/Users/Kim/Documents/trf_dir/TestDownloadFiles/converted/'
converted_local_root = '../converted/' + request['id'] + '/'
strategy_local_root = '../document/strategy/'

# # print(json.dumps(request_dict, indent=4, sort_keys=True))

project_id = request['project_id']
project_name = request['project_name']
criteria = request['criteria']

print('========== Beginning file download with urllib2. ==========')
# to_process_files = []
doc_path_dict_ = {}
undownload_docs = []
for doc_id, document in request['documents'].items():
    # print('document id: {0}'.format(doc_id))
    # print(document)

    url = document['url']
    file = Util.path_leaf(url)
    abs_file_path = input_local_root + file
    # print(abs_file_path)

    if not os.path.isfile(abs_file_path):
        try:
            print('downloading file from this url: \"{0}\" with this file name : \"{1}\".'.format(url, file))
            urllib.request.urlretrieve(url, abs_file_path)
            doc_path_dict_[doc_id] = abs_file_path
        except:
            print('An exception occurred when downloading a file from this url, \"{0}\"'.format(url))
            # Record this document that cannot be downloaded in an error list.
            undownload_docs.append(doc_id)
    else:
        print('-- This file, \"{0}\", already exists in: \"{1}\"! Therefore, this file will not be downloaded. --'.format(file, input_local_root))
        doc_path_dict_[doc_id] = abs_file_path

sim = ""
if criteria == 0:
    sim = BagOfWordSimilarity.similarity(
        request['id'],
        input_local_root,
        converted_local_root,
        strategy_local_root,
        doc_path_dict_, project_id,
        project_name,
        undownload_docs)
    # filename = "bag_of_word_sim.json"
elif criteria == 1:
    sim = WordEmbeddedSimilarity.similarity(
        input_local_root,
        converted_local_root,
        strategy_local_root,
        doc_path_dict_,
        project_id,
        project_name,
        undownload_docs)
    # filename = "word_em_sim.json"

send_progress(id=request['id'], code="192", data=str(sim))
send_progress(id=request['id'], code="050", keep=True)
print(sim)
# with open(filename, 'w', encoding='utf-8') as outfile:
#     json.dump(str(sim), outfile, ensure_ascii=False, indent=4)
