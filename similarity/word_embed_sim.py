import sys

sys.path.append("..")
sys.path.append("../..")  # Adds higher directory to python modules path.

from service_helper import send_progress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PDFreader.pdfReader import extract_pdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import natsort
import re
from pythainlp.word_vector import *
from Util import Util

import docx2txt
from TextPreProcessing import TextPreProcessing


class WordEmbeddedSimilarity:
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
        for proj_id in data.keys():
            data_content.append(data[proj_id][0])
            for key, value in doc_path_dict.items():
                if proj_id in value:
                    name = key
            data_doc.append(name)
            file_path = doc_path_dict[name]
        data_df_dict = {'proj_id': data_doc, 'file_path': file_path, 'content': data_content}
        data_df = pd.DataFrame.from_dict(data_df_dict)
        return data_df

    @staticmethod
    def sentence_vec(words: list, use_mean: bool = True):
        vec = np.zeros((1, 300))
        _MODEL = get_model()
        # use_mean = True
        for word in words:
            if word == " ":
                word = "xxspace"
            elif word == "\n":
                word = "xxeol"

            if word in _MODEL.wv.index2word:
                vec += _MODEL.wv.word_vec(word)
            else:
                pass

        if use_mean:
            vec /= len(words)

        # print(vec)
        return vec

    @staticmethod
    def similarity(id, input_local_root, converted_local_root, strategy_local_root, doc_path_dict_, project_id,
                   project_name, undownload_docs):
        # util = Util()
        print("[Word Embed Similarity]")
        topic_sim = []
        unreadable_docs = []
        for num in range(17):
            part = num + 1
            send_progress(id=id, code="S00", payload=[part], keep=True)
            print("========== PART 1 : Input Files ==========")
            doc_path_dict = {}
            doc_path_dict = doc_path_dict_.copy()
            # set path here
            # doc_path_dict = {"RDG5430015":"document/docx/RDG5430015.docx", 
            # "RDG5430002":"document/docx/RDG5430002.docx",
            # "RDG5250070":"document/docx/RDG5250070.docx",
            # "MRG5980259":"document/docx/MRG5980259.docx",
            # "MRG5980243":"document/docx/MRG5980243.docx"}
            if num == 0:
<<<<<<< HEAD
                doc_path_dict["Policy_and_strategy"] = strategy_local_root+"Policy_and_strategy.pdf"
                strategy_doc_name = "Policy_and_strategy.pdf"
            else:
                doc_path_dict["Policy_and_strategy_only_prog"+str(num)] = strategy_local_root+"ยุทธศาสตร์_อววน_sep_programs/Policy_and_strategy_only_prog"+str(num)+".docx"
=======
                doc_path_dict["0"] = streategy_local_root+"Policy_and_strategy.docx"
                strategy_doc_name = "Policy_and_strategy.pdf"
            else:
                doc_path_dict["0"] = streategy_local_root+"ยุทธศาสตร์_อววน_sep_programs/Policy_and_strategy_only_prog"+str(num)+".docx"
>>>>>>> upstream/master
                strategy_doc_name = "Policy_and_strategy_only_prog"+str(num)+".docx"

            data, unreadable_docs = Util.find_read_file(id, doc_path_dict, converted_local_root, unreadable_docs)
            num_doc = len(data)
            num_title = len(doc_path_dict)
            # print(num_doc)

            print("========== PART 2 : Data Preparation ==========")
            if num_doc == 0 and num_title == 0:
                return
            if num_doc != num_title:
                return

            send_progress(id=id, code="S20", payload=[part])
            # Set data into dataframe type
            data_df = WordEmbeddedSimilarity.to_dataframe(data, doc_path_dict)
            # data_df.head()
            print(data_df)

            print("========== PART 3 : Creating Word Tokenize ==========")
            # Word Tokenization
            send_progress(id=id, code="S30", payload=[part])
            inp_list = []
            for num in range(num_doc):
                content = data_df['content'][num]
                words = TextPreProcessing.split_word(content)
                inp_list.append(words)

            print("========== PART 4 : Measure the Cosine Similarity ==========")
            send_progress(id=id, code="S40", payload=[part])
            doc_score = {}
            for i in range(1, len(inp_list)):
                cos = cosine_similarity(WordEmbeddedSimilarity.sentence_vec(inp_list[0]),
                                        WordEmbeddedSimilarity.sentence_vec(inp_list[i]))
                doc_score[data_df['proj_id'][i]] = cos[0][0]
            doc_ranking = {key: rank for rank, key in enumerate(sorted(doc_score, key=doc_score.get, reverse=True), 1)}
            # print(doc_score)
            # print(doc_ranking)
            send_progress(id=id, code="S50", payload=[part])
            topic_rank = []
            for proj_id in doc_ranking.keys():
                sim_dict = {
                    # "ranking":doc_ranking[proj_id],
                    "document_id":proj_id,
                    "file_path": doc_path_dict[proj_id],
                    "score": doc_score[proj_id]
                }
                topic_rank.append(sim_dict)
            # print("Sentence Similarity:",topic_rank)
            topic_sim_dict = {
                "strategy_topic": strategy_doc_name,
                "similarity_ranking_score": topic_rank
            }
            topic_sim.append(topic_sim_dict)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # print(topic_sim)
        word_em_sim = {
            "criteria": 1,
            "topic_similarity": topic_sim,
            "unreadable_documents": undownload_docs,
            "undownloadable_documents": unreadable_docs
        }

        # print(word_em_sim)
        return word_em_sim

# word_em_sim = WordEmbeddedSimilarity().similarity()
# print(word_em_sim)
