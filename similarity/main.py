from bag_of_word import BagOfWordSimilarity
from word_embed_sim import WordEmbeddedSimilarity
import json 
import os
# This package is for downloading pdf
import urllib.request
from Util import Util

# define a input local root 

input_local_root = '/Users/Kim/Documents/trf_dir/TestDownloadFiles/ori/'
# input_local_root = '/Users/dhanamon/LDAitkmitl/document/TestDownloadFiles/ori/'
converted_local_root = '/Users/Kim/Documents/trf_dir/TestDownloadFiles/converted/'
# converted_local_root = '/Users/dhanamon/LDAitkmitl/document/TestDownloadFiles/converted/'
streategy_local_root = '/Users/Kim/Documents/trf_dir/strategy/'
# streategy_local_root = '/Users/dhanamon/LDAitkmitl/document/strategy/'

with open('similarity_json_request.json', 'r') as f:
    request_dict = json.load(f)

# # print(json.dumps(request_dict, indent=4, sort_keys=True))

for request in request_dict:
    documents = request['documents']

project_id = request['project_id']
project_name = request['project_name']
criteria = request['criteria']

print('========== Beginning file download with urllib2. ==========')
# to_process_files = []
doc_path_dict_ = {}
undownload_docs = []
for doc_id, document in documents.items():
    # print('document id: {0}'.format(doc_id))
    # print(document)

    url = document['url']
    file = Util.path_leaf(url)
    abs_file_path =  input_local_root + file
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


if criteria == 0:
    sim = BagOfWordSimilarity.similarity(input_local_root,converted_local_root,streategy_local_root,doc_path_dict_,project_id,project_name,undownload_docs)
    filename = "bag_of_word_sim.json"
elif criteria == 1:
    sim = WordEmbeddedSimilarity.similarity(input_local_root,converted_local_root,streategy_local_root,doc_path_dict_,project_id,project_name,undownload_docs)
    filename = "word_em_sim.json"

with open(filename, 'w', encoding='utf-8') as outfile:
    json.dump(str(sim), outfile, ensure_ascii=False, indent=4)