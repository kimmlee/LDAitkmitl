import sys
sys.path.append("..")

from Util import Util
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
        'projectId': sys.argv[2],
        'documents': ast.literal_eval(sys.argv[3]),
        'max_no_topic': int(sys.argv[4])
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

# print(json.dumps(request_dict, indent=4, sort_keys=True))

print('========== Beginning file download with urllib2. ==========')
to_process_files = []
to_process_titles = []
error_doc_ids = []
counter = 0

send_progress(
    id=request['id'],
    code="011",
    payload=["s" if len(to_process_files) > 0 else ""],
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
            to_process_titles.append(document['title'])

            send_progress(
                id=request['id'],
                code="021",
                payload=[file])

        except:
            print('An exception occurred when downloading a file from this url, \"{0}\"'.format(url))
            # Delete this document that cannot be downloaded at a specific index.
            error_doc_ids.append(doc_id)

            send_progress(
                id=request['id'],
                code="410",
                payload=[url],
                keep=True)

    else:
        print('-- This file, \"{0}\", already exists in: \"{1}\"! Therefore, this file will not be downloaded. --'.format(file, input_local_root))
        to_process_files.append(file)
        to_process_titles.append(document['title'])

    counter += 1

# print('========================')
# print(documents)

ldamodeling = LDAModeling()
ldamodeling.perform_topic_modeling(request['id'], input_local_root, to_process_files, to_process_titles, converted_local_root,
                                   output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file,
                                   request['max_no_topic'])

# max_no_topic = 10
#
# print('========== Beginning file download with urllib2. ==========')
# to_process_files = []
# abs_file_paths = []
# counter = 0
# #print(len(urls), len(titles))
# for url in urls:
#     file = Util.path_leaf(url)
#     # print(file_)
#     abs_file_path =  input_local_root + file
#     # print(abs_file_path)
#
#     if not os.path.isfile(abs_file_path):
#         try:
#             print('downloading file from this url: \"{0}\" with this file name : \"{1}\".'.format(url, file))
#             urllib.request.urlretrieve(url, abs_file_path)
#         except:
#             print('An exception occurred when downloading a file from this url, \"{0}\"'.format(url))
#             # Delete the title of a file that cannot be downloaded at a specific index.
#             # This is to keep two lists of abs_file_paths and titles consistent.
#
#             del titles[counter]
#
#     else:
#         print('-- This file, \"{0}\", already exists in: \"{1}\"! Therefore, this file will not be downloaded. --'.format(file, input_local_root))
#     to_process_files.append(file)
#     counter += 1
#
# ldamodeling = LDAModeling()
# ldamodeling.perform_topic_modeling(input_local_root, to_process_files, titles, converted_local_root,
#                                    output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file,
#                                    max_no_topic)


