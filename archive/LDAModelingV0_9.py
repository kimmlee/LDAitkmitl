import sys
sys.path.append("..") # Adds higher directory to python modules path.

from Util import Util
from TextPreProcessing import TextPreProcessing
from TextDistribution import TextDistribution

from pythainlp import word_tokenize
from pythainlp.corpus import thai_words, thai_stopwords, thai_syllables
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pyLDAvis.gensim
from gensim import corpora, models
from gensim.models import LsiModel, LdaModel, CoherenceModel
from pylexto import LexTo

import string
import pythainlp.corpus

import random
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt

import os
import bs4
import re

# This package is for downloading pdf
import urllib.request


# todo edit "Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes"
class LDAModeling:

    def __init__(self):
        self.num_cut = 2

    def to_dataframe(self, data, titles):
        """
        Changing document in dictionary to dataframe and setting field like...
        | doc_id | title | content |

        doc_id: Document's file name.
        title: Tile of document.
        content: Content of document.
        """
        data_doc = []
        data_titles = titles
        data_content = []
        for doc_id in data.keys():
            data_content.append(data[doc_id][0])
            data_doc.append(doc_id)
        data_df_dict = {'doc_id': data_doc, 'title': data_titles, 'content': data_content}
        data_df = pd.DataFrame.from_dict(data_df_dict)
        return data_df

    def localize_pyLDAvis_to_thai(self, en_input_dir, en_pyLDAvis_file, th_output_dir, th_pyLDAvis_file):
        with open(en_input_dir + en_pyLDAvis_file) as inf:
            txt = inf.read()
            soup = bs4.BeautifulSoup(txt)

        meta = soup.new_tag("meta", charset="utf-8")
        soup.head.append(meta)

        souptemp = soup.prettify()
        souptemp = souptemp.replace('https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js', 'd3.min.js')
        souptemp = souptemp.replace('https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js', 'ldavis.v1.0.0.js')
        souptemp = souptemp.replace('https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css',
                                    'ldavis.v1.0.0.css')

        with open(th_output_dir + th_pyLDAvis_file, "w") as outf:
            outf.write(souptemp)


    """to remove"""
    # Generate LDA Model
    def LDAmodel(self, dictionary, corpus, num_top=10):
        ldamodel = LdaModel(corpus, num_top, id2word=dictionary, decay=0.6, random_state=2, passes=10)
        return ldamodel

    """to remove"""
    # Generate LSI Model
    def LSImodel(self, dictionary, corpus_tfidf, num_top=6):
        lsimodel = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_top, decay=0.5)
        return lsimodel

    def perform_topic_modeling(self, input_local_root, files, titles,
                               output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file,
                               max_no_topics = 10, is_short_words_removed = True):

        print("========== PART 1 : Input Files ==========")
        data = Util.filter_file_to_read(input_local_root, files)
        num_doc = len(titles)

        print("========== PART 2 : Data Preparation and Creating Word Tokenization ==========")
        # Set data into dataframe type
        data_df = self.to_dataframe(data, titles)
        data_df.head()

        inp_list = []
        for num in range(num_doc):
            content = data_df['content'][num]
            inp_list.append(TextPreProcessing.split_word(content))

        counter = 0
        for word in inp_list:
            counter += len(word)
        print("Unique words in this processing corpus: {0}".format(counter))

        # Create dictionary, corpus and corpus TFIDF
        # Turn tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(inp_list)
        dict2 = {dictionary[ID]:ID for ID in dictionary.keys()}

        # Convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in inp_list]
        tfidf = models.TfidfModel(corpus, smartirs='ntc')
        corpus_tfidf = tfidf[corpus]

        if is_short_words_removed:
            # Remove character number is less than 2 words off
            new_lists = TextPreProcessing.cut_character(inp_list, self.num_cut)
        else:
            new_lists = inp_list

        # Remove word is not noun and prop noun by pos_tag function
        for num in range(num_doc):
            new_lists[num] = TextPreProcessing.postag(new_lists[num])

        # Create new dict and corpus
        dictionary2 = corpora.Dictionary(new_lists)
        dict_2 = {dictionary2[ID]:ID for ID in dictionary2.keys()}
        corpus2 = [dictionary2.doc2bow(text) for text in new_lists]

        # Header Title plus frequency in corpus
        corpus2 = TextPreProcessing.add_frequency(dict_2, corpus2, data_df, 10, num_doc)

        print("========== PART 3 : Generate LDA Model ==========")
        # Generate LDA Model


        # Default number of topic is 10. If the number of documents is fewer than the maximum number of topics, the number of documents will be used to as the maximum number of topics.
        # if num_doc <= 10:
        #     max_no_topics = num_doc
        max_no_topics = min([max_no_topics, num_doc])

        ldamodel = self.LDAmodel(dictionary2, corpus2, max_no_topics)
        term_dist_topic = ldamodel.show_topics(max_no_topics, 1000, log=True, formatted=False)
        print(term_dist_topic)

        print("========== PART 4 : Topic-term distribution ==========")
        ### Topic-Term Dist
        topic_term_dist = []
        topic_term_dist = TextDistribution.topicTerm_dist(topic_term_dist, term_dist_topic)
        print(topic_term_dist)

        print("========== PART 4-1 : Document-topic (all) distribution ==========")
        ### Doc_topic_all_dist
        doc_topic_dist = []
        doc_topic_dist = TextDistribution.docTopic_dist(doc_topic_dist, data_df, num_doc, inp_list,dictionary2,ldamodel)
        print(doc_topic_dist)

        print("========== PART 4-2 : Topic-term (min) distribution ==========")
        ### Doc_topic_min_dist
        n_doc_intopic = []
        n_doc_intopic = TextDistribution.Ndoc_topic(n_doc_intopic,num_doc, data_df, inp_list, dictionary2, ldamodel)
        print(n_doc_intopic)


        #lsimodel = self.LSImodel(dictionary2, corpus2, 6)

        # lsimodel.show_topics(6, 10, log=True, formatted=False)

        print("========== PART 5 : Evaluate Model ==========")
        # Evaluate
        #lsi_coherence = CoherenceModel(lsimodel, corpus=corpus_tfidf, dictionary=dictionary2, coherence='u_mass')
        lda_coherence = CoherenceModel(ldamodel, corpus=corpus2, dictionary=dictionary2, coherence='u_mass')
        print(lda_coherence.get_coherence_per_topic())
        # print("LDA umass score = %.4f , LSI umass score = %.4f"% (lda_coherence.get_coherence(),
        #                                                           lsi_coherence.get_coherence()))
        print("LDA umass score = %.4f" % (lda_coherence.get_coherence()))

        #lsi_coherence = CoherenceModel(lsimodel, texts=new_lists, dictionary=dictionary2, coherence='c_uci')
        lda_coherence = CoherenceModel(ldamodel, texts=new_lists, dictionary=dictionary2, coherence='c_uci')
        # print("LDA uci score = %.4f , LSI uci score = %.4f"% (lda_coherence.get_coherence(),
        #                                                       lsi_coherence.get_coherence()))
        print("LDA uci score = %.4f" % (lda_coherence.get_coherence()))

        print("========== PART 6 : Export pyLDAvis HTML ==========")
        # pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(ldamodel, corpus2, dictionary=ldamodel.id2word)
        pyLDAvis.save_html(vis, output_dir + pyLDAvis_output_file)

        print("========== PART 7 : Convert pyLDAvis HTML to Thai==========")
        self.localize_pyLDAvis_to_thai(output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file)