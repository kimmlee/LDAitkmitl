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
from pythainlp.word_vector import *

import docx2txt
from TextPreProcessing import split_word, cut_character, postag, add_frequency


class Util:
    def read_file(self, files):
        data_file =[]
        data={}
        for f in files:
            try:
                f_list = re.split("; |/|\\.", f)
                if f.endswith('.pdf'):
                    """
                        need modify here when the original file cannot be read.
                    """
                    data_file_text = extract_pdf(f)
                elif f.endswith('.docx'):
                    data_file_text = docx2txt.process(f)

                data_file.append(data_file_text)
            except:
                print("=======ERROR cannot fild the below file in a given path=======")
                print(f, f_list)
                print('+++++++++++++++++++')

            # Add Topic in dict
            # try:
            #     print(f)
            #     for key in self.data.keys():
            #         print("------->",key)
            # except:
            #     pass
            data[f_list[-2]] = [str(data_file_text)]
        return data

    def find_read_file(self, doc_path_dict):
        """
        Finding file in input path file and keep in dictionary
        """
        files = []
        for proj_id in doc_path_dict.keys():
            files.append(doc_path_dict[proj_id])
        files = natsort.natsorted(files,reverse=True)
        print(files)
        #Find file in folder path and extract file to text
        data = self.read_file(files)
        return data

class WordEmbeddedSimilarity:
    def to_dataframe(self, data, doc_path_dict):
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
            data_doc.append(proj_id)
            file_path = doc_path_dict[proj_id]
        data_df_dict = {'proj_id': data_doc, 'file_path':file_path,'content': data_content}
        data_df = pd.DataFrame.from_dict(data_df_dict)
        return data_df

    def sentence_vec(self, words: list, use_mean: bool = True):
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

    def similarity(self):
        util = Util()
        topic_sim = []
        for num in range(17):
            print("========== PART 1 : Input Files ==========")
            #set path here
            doc_path_dict = {"RDG5430015":"document/docx/RDG5430015.docx", 
            "RDG5430002":"document/docx/RDG5430002.docx",
            "RDG5250070":"document/docx/RDG5250070.docx",
            "MRG5980259":"document/docx/MRG5980259.docx",
            "MRG5980243":"document/docx/MRG5980243.docx"}
            if num == 0:
                doc_path_dict["ยุทธศาสตร์_อววน_v12_ไม่มีผนวก"] = "document/strategy/ยุทธศาสตร์_อววน_v12_ไม่มีผนวก.docx"
                strategy_doc_name = "ยุทธศาสตร์_อววน_v12_ไม่มีผนวก"
            else:
                doc_path_dict["ยุทธศาสตร์_อววน_only_prog"+str(num)] = "document/strategy/ยุทธศาสตร์_อววน_sep_programs/ยุทธศาสตร์_อววน_only_prog"+str(num)+".docx"
                strategy_doc_name = "ยุทธศาสตร์_อววน_only_prog"+str(num)

            data = util.find_read_file(doc_path_dict)
            num_doc = len(data)
            # print(num_doc)

            print("========== PART 2 : Data Preparation ==========")
            # Set data into dataframe type
            data_df = self.to_dataframe(data, doc_path_dict)
            # data_df.head()
            print(data_df)

            print("========== PART 3 : Creating Word Tokenize ==========")
            # Word Tokenization
            inp_list = []
            for num in range(num_doc):
                content = data_df['content'][num]
                words = split_word(content)
                inp_list.append(words)

            doc_score = {}
            for i in range(1,len(inp_list)):
                cos = cosine_similarity(self.sentence_vec(inp_list[0]),self.sentence_vec(inp_list[i]))
                doc_score[data_df['proj_id'][i]] = cos[0][0]
            doc_ranking = {key: rank for rank, key in enumerate(sorted(doc_score, key=doc_score.get, reverse=True), 1)}
            # print(doc_score)
            # print(doc_ranking)

            topic_rank = []
            for proj_id in doc_ranking.keys():
                sim_dict = {"ranking":doc_ranking[proj_id],
                            "proj_proposal_id":proj_id,
                            "file_path": doc_path_dict[proj_id],
                            "score":doc_score[proj_id]}
                topic_rank.append(sim_dict)
            # print("Sentence Similarity:",topic_rank)
            topic_sim_dict = {
                "Strategy Topic": strategy_doc_name,
                "Similarity Ranking Score":topic_rank
            }
            topic_sim.append(topic_sim_dict)

        # print(topic_sim)
        word_em_sim = {
            "Similarity Type":"Word Embedding Similarity",
            "Topic Similarity": topic_sim
        }
        # print(word_em_sim)
        return word_em_sim


word_em_sim = WordEmbeddedSimilarity().similarity()
print(word_em_sim)

