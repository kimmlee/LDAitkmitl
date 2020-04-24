import sys
sys.path.append("..")

from Util import Util
#from LDAModeling import LDAModeling
from LDAModeling import LDAModeling
import os
import urllib.request
import json
import ast
from service_helper import send_progress

as_worker = True

if not as_worker:
    with open('json_request.json', 'r') as f:
        request = json.load(f)
else:
    request = {
        'id': sys.argv[1],
        'project_id': sys.argv[2],
        'project_name': sys.argv[3],
        'documents': ast.literal_eval(sys.argv[4]),
        'max_no_topic': int(sys.argv[5])
    }

"""
    1) download all files from a list of URLs and save to local (API server)
    require:
        local_path = "/Users/Kim/Documents/TestDownloadFiles/"
"""
# todo change these three variables
# define a local root to save files

input_local_root = './documents/' + request['id'] + '/'
# input_local_root = '/Users/dhanamon/LDAitkmitl/TestDownloadFiles/'
# input_local_root = '/Users/dhanamon/Google Drive/TRF_Y61_Mon/05012020_Topic modeling/document/TestDownloadFiles/ori/'

converted_local_root = './converted/' + request['id'] + '/'

# define an output directory to save an 'original' pyLDAvis html file
output_dir = './results/'

# output_dir = '/Users/dhanamon/LDAitkmitl/PyLDAVizOutput/'
# define an output directory to save an 'original' pyLDAvis html file
pyLDAvis_output_file = 'en-result-' + str(request['id']) + '.html'

# define an output directory to save an 'thai' pyLDAvis html file
th_output_dir = './results/'

# th_output_dir = '/Users/dhanamon/LDAitkmitl/PyLDAVizOutput/th/'
# define an output directory to save an 'thai' pyLDAvis html file
th_pyLDAvis_output_file = 'th-result-' + str(request['id']) + '.html'

# with open('json_request.json', 'r') as f:
#     request_dict = json.load(f)

# print(json.dumps(request_dict, indent=4, sort_keys=True))

id = request['id']
project_id = request['project_id']
project_name = request['project_name']
max_no_topic = int(request['max_no_topic'])

print('========== Beginning file download with urllib2. ==========')

to_process_files = []
to_process_titles = []
doc_path_file = {}
undownload_docs = []

send_progress(
    id=id,
    code="011",
    payload=["s" if len(request['documents']) > 1 else ""],
    keep=True)

for doc_id, document in request['documents'].items():
    # print('document id: {0}'.format(doc_id))
    # print(document)

    url = document['url']
    file = Util.path_leaf(url)
    # print(file_)
    abs_file_path = input_local_root + file
    # print(abs_file_path)

    if not os.path.isfile(abs_file_path):
        try:
            print('downloading file from this url: \"{0}\" with this file name : \"{1}\".'.format(url, file))
            urllib.request.urlretrieve(url, abs_file_path)

            to_process_files.append(file)
            doc_path_file[doc_id] = abs_file_path
            to_process_titles.append(document['title'])
            send_progress(
                id=id,
                code="021",
                payload=[file])
        except:
            print('An exception occurred when downloading a file from this url, \"{0}\"'.format(url))
            # Record this document that cannot be downloaded in an error list.
            undownload_docs.append(doc_id)
            send_progress(
                id=id,
                code="410",
                payload=[url],
                keep=True)
    else:
        print('-- This file, \"{0}\", already exists in: \"{1}\"! Therefore, this file will not be downloaded. --'.format(file, input_local_root))
        to_process_files.append(file)
        doc_path_file[doc_id] = abs_file_path
        to_process_titles.append(document['title'])

# print('========================')
# print(documents)

ldamodeling = LDAModeling()
ldamodeling.perform_topic_modeling(
    id, 
    project_id, 
    project_name, 
    input_local_root, 
    to_process_files, 
    to_process_titles,
    converted_local_root,
    doc_path_file,
    output_dir, 
    pyLDAvis_output_file, 
    th_output_dir, 
    th_pyLDAvis_output_file, 
    undownload_docs,
    max_no_topic)

# max_no_topic = 10

# print('========== Beginning file download with urllib2. ==========')
# to_process_files = []
# counter = 0
# #print(len(urls), len(titles))
# for url in urls:
#     file = Util.path_leaf(url)
#     # print(file_)
#     abs_file_path =  input_local_root + file
#     # print(abs_file_path)

#     if not os.path.isfile(abs_file_path):
#         try:
#             print('downloading file from this url: \"{0}\" with this file name : \"{1}\".'.format(url, file))
#             urllib.request.urlretrieve(url, abs_file_path)
#         except:
#             print('An exception occurred when downloading a file from this url, \"{0}\"'.format(url))
#             # Delete the title of a file that cannot be downloaded at a specific index.
#             # This is to keep two lists of to_process_files and titles consistent.
#             del titles[counter]
#     else:
#         print('-- This file, \"{0}\", already exists in: \"{1}\"! Therefore, this file will not be downloaded. --'.format(file, input_local_root))
#     to_process_files.append(file)
#     counter += 1

# project_name = "โครงการวิเคราะห์ xxx"

# ldamodeling = LDAModeling()
# # ldamodeling.perform_topic_modeling(input_local_root, to_process_files, titles, converted_local_root,
# #                                    output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file,
# #                                    max_no_topic)

# result = ldamodeling.perform_topic_modeling(project_name, input_local_root, to_process_files, titles, converted_local_root,
#                                    output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file,
#                                    max_no_topic)

# # result['project_id'] = project_id
# # result['undownloadable_documents'] = undownload_docs

# with open('result.json', 'w', encoding='utf-8') as outfile:
#     json.dump(str(result), outfile, ensure_ascii=False, indent=4)