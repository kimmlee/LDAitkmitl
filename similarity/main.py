from bag_of_word import BagOfWordSimilarity
from word_embed_sim import WordEmbeddedSimilarity
import json 
import os
# This package is for downloading pdf
import urllib.request
from Util import Util

# define a local root to save files

# input_local_root = '/Users/Kim/Documents/trf_dir/TestDownloadFiles/ori/'
input_local_root = '/Users/dhanamon/LDAitkmitl/document/TestDownloadFiles/ori/'
# converted_local_root = '/Users/Kim/Documents/trf_dir/TestDownloadFiles/converted/'
converted_local_root = '/Users/dhanamon/LDAitkmitl/document/TestDownloadFiles/converted/'
streategy_local_root = '/Users/dhanamon/LDAitkmitl/document/strategy/'

with open('similarity_json_request.json', 'r') as f:
    request_dict = json.load(f)

# # print(json.dumps(request_dict, indent=4, sort_keys=True))

for request in request_dict:
    documents = request['documents']

project_id = request['project_id']
project_name = request['project_name']

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

# bag_of_word_sim = BagOfWordSimilarity.similarity(input_local_root,converted_local_root,streategy_local_root,doc_path_dict_,project_id,project_name,undownload_docs)
# # print(bag_of_word_sim)

# with open('bag_of_word_sim.json', 'w', encoding='utf-8') as outfile:
#     json.dump(str(bag_of_word_sim), outfile, ensure_ascii=False, indent=4)


word_em_sim = WordEmbeddedSimilarity.similarity(input_local_root,converted_local_root,streategy_local_root,doc_path_dict_,project_id,project_name,undownload_docs)

with open('word_em_sim.json', 'w', encoding='utf-8') as outfile:
    json.dump(str(word_em_sim), outfile, ensure_ascii=False, indent=4)