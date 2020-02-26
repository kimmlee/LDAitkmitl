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

import docx2txt
from TextPreProcessing import split_word, cut_character, postag, add_frequency

# Create a new Utility class to for file management and loading
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


class BagOfWordSimilarity:
    # Set data into dataframe type
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
        print(doc_path_dict)
        for proj_id in data.keys():
            print(proj_id)
            data_content.append(data[proj_id][0])
            data_doc.append(proj_id)
            # print(proj_id)
            # print(doc_path_dict[proj_id])
            file_path = doc_path_dict[proj_id]
        data_df_dict = {'proj_id': data_doc, 'file_path':file_path,'content': data_content}
        self.data_df = pd.DataFrame.from_dict(data_df_dict)
        return self.data_df

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
            # print("---------------->",num)
            if num == 0:
                doc_path_dict["ยุทธศาสตร์_อววน_v12_ไม่มีผนวก"] = "document/strategy/ยุทธศาสตร์_อววน_v12_ไม่มีผนวก.docx"
                strategy_doc_name = "ยุทธศาสตร์_อววน_v12_ไม่มีผนวก"
            else:
                doc_path_dict["ยุทธศาสตร์_อววน_only_prog"+str(num)] = "document/strategy/ยุทธศาสตร์_อววน_sep_programs/ยุทธศาสตร์_อววน_only_prog"+str(num)+".docx"
                strategy_doc_name = "ยุทธศาสตร์_อววน_only_prog"+str(num)
            
            data = util.find_read_file(doc_path_dict)
            # for key in data.keys():
            #     print(key)
            num_doc = len(data)
            print("========== PART 2 : Data Preparation ==========")
            data_df = self.to_dataframe(data, doc_path_dict)
            # data_df.head()
            print(data_df)

            print("========== PART 3 : Creating Word Tokenize ==========")
            # Word Tokenization
            inp_list = []
            for num in range(num_doc):
                content = data_df['content'][num]
                inp_list.append(split_word(content))

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

            topic_rank = []
            for proj_id in doc_ranking.keys():
                sim_dict = {"ranking":doc_ranking[proj_id],
                            "proj_proposal_id":proj_id,
                            "file_path": doc_path_dict[proj_id],
                            "score":doc_score[proj_id]}
                topic_rank.append(sim_dict)
            # print("Bag of word Similarity:",topic_rank)
            topic_sim_dict = {
                "Strategy Topic": strategy_doc_name,
                "Similarity Ranking Score":topic_rank
            }
            topic_sim.append(topic_sim_dict)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # print(topic_sim)
        bag_of_word_sim = {
            "Similarity Type":"Bag of word similarity",
            "Topic Similarity": topic_sim
        }
        # print(bag_of_word_sim)
        return bag_of_word_sim

bag_of_word_sim = BagOfWordSimilarity().similarity()
print(bag_of_word_sim)





