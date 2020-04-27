import sys
sys.path.append("..") # Adds higher directory to python modules path.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PDFreader.pdfReader import extract_pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import natsort 
import re
import json
import os
# This package is for downloading pdf
import urllib.request
import ntpath
from pdfminer.pdfparser import PDFSyntaxError
from Util import Util

import docx2txt
from TextPreProcessing import TextPreProcessing

class BagOfWordSimilarity:
    # Set data into dataframe type
    @staticmethod
    def to_dataframe(data, doc_path_dict):
        """
        Changing document in dictionary to dataframe and setting field like...
        | proj_id | file_path | content |

        proj_id: Document's file name.
        file_path: File path of document.
        content: Content of document.
        """
        data_doc = []
        # data_title = title
        data_content = []
        # print(doc_path_dict)
        for proj_id in data.keys():
            data_content.append(data[proj_id][0])
            for key,value in doc_path_dict.items():
                if proj_id in value:
                    name = key
            data_doc.append(name)
            file_path = doc_path_dict[name]
        data_df_dict = {'proj_id': data_doc, 'file_path':file_path,'content': data_content}
        data_df = pd.DataFrame.from_dict(data_df_dict)
        return data_df

    @staticmethod
    def similarity(input_local_root,converted_local_root,streategy_local_root,doc_path_dict_,project_id,project_name, undownload_docs):
        topic_sim = []
        unreadable_docs = []
        for num in range(1):
            doc_path_dict = {}
            doc_path_dict = doc_path_dict_.copy()
            print("========== PART 1 : Input Files ==========")
            if num == 0:
                doc_path_dict["0"] = streategy_local_root+"Policy_and_strategy.docx"
                strategy_doc_name = "Policy_and_strategy.pdf"
            else:
                doc_path_dict["0"] = streategy_local_root+"ยุทธศาสตร์_อววน_sep_programs/Policy_and_strategy_only_prog"+str(num)+".docx"
                strategy_doc_name = "Policy_and_strategy_only_prog"+str(num)+".docx"
            # print(doc_path_dict)
            data, unreadable_docs = Util.find_read_file(doc_path_dict, converted_local_root, unreadable_docs)
            # for key in data.keys():
            #     print(key)
            num_doc = len(data)
            num_title = len(doc_path_dict)
            print("========== PART 2 : Data Preparation ==========")
            if num_doc == 0 and num_title == 0:
                bag_of_word_sim = {
                    "project_id":project_id,
                    "project_name":project_name,
                    "success":False,
                    "errorMessage":None,
                    "criteria":0,
                    "topic_similarity": None,
                    "undownload_docs":undownload_docs,
                    "unreadable_docs":unreadable_docs
                }
                return bag_of_word_sim
            if num_doc != num_title:
                bag_of_word_sim = {
                    "project_id":project_id,
                    "project_name":project_name,
                    "success":False,
                    "errorMessage":None,
                    "criteria":0,
                    "topic_similarity": None,
                    "undownload_docs":undownload_docs,
                    "unreadable_docs":unreadable_docs
                }
                return bag_of_word_sim

            data_df = BagOfWordSimilarity.to_dataframe(data, doc_path_dict)
            # data_df.head()
            print(data_df)

            print("========== PART 3 : Creating Word Tokenize ==========")
            # Word Tokenization
            inp_list = []
            for num in range(num_doc):
                content = data_df['content'][num]
                inp_list.append(TextPreProcessing.split_word(content))

            print("========== PART 4 : Term Weighting with TfidfVectorizer ==========")
            # Term Weighting with TfidfVectorizer
            inp_list_j = [','.join(tkn) for tkn in inp_list]
            # print(inp_list_j)
            tvec = TfidfVectorizer(analyzer=lambda x:x.split(','),)
            t_feat = tvec.fit_transform(inp_list_j)
            # print(t_feat)

            print("========== PART 5 : Measure the Cosine Similarity ==========")
            doc_score = {}
            # Measure the cosine similarity between the first document vector and all of the others
            max_cos = 0
            best_row = 0
            for row in range(1,t_feat.shape[0]):
                cos = cosine_similarity(t_feat[0], t_feat[row])
                # print(row, cos)
                doc_score[data_df['proj_id'][row]] = cos[0][0]
                # best so far?
                if cos > max_cos:
                    max_cos = cos
                    best_row = row
            print("Most similar document was row %d: cosine similarity = %.3f" % ( best_row, max_cos ) )
            # Best document - just display the start of it
            # print(doc_score)
            doc_ranking = {key: rank for rank, key in enumerate(sorted(doc_score, key=doc_score.get, reverse=True), 1)}
            # print(doc_ranking)

            # doc_ranking will sort score value from high to low
            topic_rank = []
            for proj_id in doc_ranking.keys():
                sim_dict = {
                    # "ranking":doc_ranking[proj_id],
                    "document_id":proj_id,
                    "file_path": doc_path_dict[proj_id],
                    "score":doc_score[proj_id]
                }
                topic_rank.append(sim_dict)
            # print("Bag of word Similarity:",topic_rank)
            topic_sim_dict = {
                "strategy_topic": strategy_doc_name,
                "similarity_ranking_score":topic_rank
            }
            topic_sim.append(topic_sim_dict)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # print(topic_sim)
        bag_of_word_sim = {
            "project_id":project_id,
            "project_name":project_name,
            "success":True,
            "errorMessage":None,
            "criteria":0,
            "topic_similarity": topic_sim,
            "undownload_docs":undownload_docs,
            "unreadable_docs":unreadable_docs
        }
        # print(bag_of_word_sim)
        return bag_of_word_sim







