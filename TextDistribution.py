import math
import numpy as np
# from __future__ import absolute_import
import funcy as fp
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import issparse
from past.builtins import xrange, basestring
import gensim
from collections import namedtuple
import json
import logging


class TextDistribution:

    """
    A TextDistribution class is to find the text statistics of input corpus such as Topic-Term Distribution, Document distribution etc. 
    This class contains 10 static methods, including: 
        1) topicTerm_dist()
        2) _extract_data()
        3) _df_with_names()
        4) _series_with_name()
        5) _find_relevance()
        6) document_dist()
        7) docTopic_dist()
        8) document_dist_min()
        9) Ndoc_topic()
        10) compute_term_pairs()
    """

    @staticmethod
    def topicTerm_dist(ldamodel, corpus):
        """
        Topic-Term Distribution is a probability score of word in topic which display on pyLDAvis output.
        This method is called 4 methods in class for calculating the probability that extend from pyLDAvis method, including:
            1) _extract_data(): generate an extracted data by ldamodel, corpus and dictionary, return a dictionary which contains
                topic_term_dists, doc_topic_dicts, term_frequency, doc_lengths, and vocab.
            2) _df_with_names(): reformat Dataframe, use with topic_term_dists and doc_topic_dicts.
            3) _series_with_name(): reformat Series, use with term_frequency, doc_length, and vocab.
            4) _find_relevance(): calculate relevance by list of topic, vocab, log_ttd, log_lift, and lambda.

        Parameters
        ----------
        ldamodel: a topic model, required.
        corpus: a copus list of multiplied frequency of document's title with defined number, required.
            
        Returns
        ----------
        topic_term_dist: a list of dictionaries which contain topics and 1000 terms with their score in lambda (λ) = 0.6
            [
                {
                    'topic_id': 0, 
                    'terms': [
                        {
                            'term': 'ทุจริต', 
                            'score': 203.54537082463503
                        },...
                    ]
                },...
            ]
        """
        extract_data = TextDistribution._extract_data(ldamodel, corpus, dictionary=ldamodel.id2word)
        topic_term_dists = TextDistribution._df_with_names(extract_data['topic_term_dists'], 'topic', 'term')
        doc_topic_dists  = TextDistribution._df_with_names(extract_data['doc_topic_dists'], 'doc', 'topic')
        term_frequency   = TextDistribution._series_with_name(extract_data['term_frequency'], 'term_frequency')
        doc_lengths      = TextDistribution._series_with_name(extract_data['doc_lengths'], 'doc_length')
        vocab            = TextDistribution._series_with_name(extract_data['vocab'], 'vocab')
        topic_freq       = (doc_topic_dists.T * doc_lengths).T.sum()
        topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
        topic_list = list(topic_proportion.index)
        term_proportion = term_frequency / term_frequency.sum()
        log_lift = np.log(topic_term_dists / term_proportion)
        log_ttd = np.log(topic_term_dists)
        topic_term_dist = TextDistribution._find_relevance(topic_list, vocab,log_ttd, log_lift, 1000, lambda_=0.6)
         
        return topic_term_dist
    
    @staticmethod
    def _extract_data(topic_model, corpus, dictionary, doc_topic_dists=None):

        if not gensim.matutils.ismatrix(corpus):
            corpus_csc = gensim.matutils.corpus2csc(corpus, num_terms=len(dictionary))
        else:
            corpus_csc = corpus
            # Need corpus to be a streaming gensim list corpus for len and inference functions below:
            corpus = gensim.matutils.Sparse2Corpus(corpus_csc)

        vocab = list(dictionary.token2id.keys())
        # TODO: add the hyperparam to smooth it out? no beta in online LDA impl.. hmm..
        # for now, I'll just make sure we don't ever get zeros...
        beta = 0.01
        fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)
        term_freqs = corpus_csc.sum(axis=1).A.ravel()[fnames_argsort]
        term_freqs[term_freqs == 0] = beta
        doc_lengths = corpus_csc.sum(axis=0).A.ravel()
        
        assert term_freqs.shape[0] == len(dictionary), 'Term frequencies and dictionary have different shape {} != {}'.format(term_freqs.shape[0], len(dictionary))
        assert doc_lengths.shape[0] == len(corpus), 'Document lengths and corpus have different sizes {} != {}'.format(doc_lengths.shape[0], len(corpus))
        
        if hasattr(topic_model, 'lda_alpha'):
            num_topics = len(topic_model.lda_alpha)
        else:
            num_topics = topic_model.num_topics
            
        if doc_topic_dists is None:
            # If its an HDP model.
            if hasattr(topic_model, 'lda_beta'):
                gamma = topic_model.inference(corpus)
            else:
                gamma, _ = topic_model.inference(corpus)
            doc_topic_dists = gamma / gamma.sum(axis=1)[:, None]
        else:
            if isinstance(doc_topic_dists, list):
                doc_topic_dists = gensim.matutils.corpus2dense(doc_topic_dists, num_topics).T
            elif issparse(doc_topic_dists):
                doc_topic_dists = doc_topic_dists.T.todense()
            doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)

        assert doc_topic_dists.shape[1] == num_topics, 'Document topics and number of topics do not match {} != {}'.format(doc_topic_dists.shape[1], num_topics)
        # get the topic-term distribution straight from gensim without
        # iterating over tuples
        
        if hasattr(topic_model, 'lda_beta'):
            topic = topic_model.lda_beta
        else:
            topic = topic_model.state.get_lambda()
        topic = topic / topic.sum(axis=1)[:, None]
        topic_term_dists = topic[:, fnames_argsort]
        
        assert topic_term_dists.shape[0] == doc_topic_dists.shape[1]
        
        return {'topic_term_dists': topic_term_dists, 'doc_topic_dists': doc_topic_dists,
            'doc_lengths': doc_lengths, 'vocab': vocab, 'term_frequency': term_freqs}
    @staticmethod
    def _df_with_names(data, index_name, columns_name):
        if type(data) == pd.DataFrame:
            # we want our index to be numbered
            df = pd.DataFrame(data.values)
        else:
            df = pd.DataFrame(data)
        df.index.name = index_name
        df.columns.name = columns_name
        return df

    @staticmethod
    def _series_with_name(data, name):
        if type(data) == pd.Series:
            data.name = name
            # ensures a numeric index
            return data.reset_index()[name]
        else:
            return pd.Series(data, name=name)

    @staticmethod
    def _find_relevance(topic_list, vocab,log_ttd, log_lift, R=1000, lambda_=0.6):
        """
        _find_relevance calculate a probability score of word in topic which display on pyLDAvis output and plus 
        return water mask in terms 
        
        """
        relevance = lambda_ * log_ttd + (1 - lambda_) * log_lift
        id_ = relevance.T.apply(lambda s: s.sort_values(ascending=False).index).head(R)
        relevance_ = relevance.T.apply(lambda s: s.sort_values(ascending=False).values).head(R)

        topic_term_dist = []
        x=1
        for num_topic in topic_list:
            term_list = []
            for num_term in range(len(id_)):
                vocab_ = vocab[id_[num_topic][num_term]]
                r_score =  relevance_[num_topic][num_term]
                term = {"term":vocab_,
                        "score":r_score}
                # ----- start water mask -----
                if num_term in [177,288,399]:
                    score_next = relevance_[num_topic][num_term+1]
                    score_wm = ((r_score - score_next)/2)+score_next
                    if num_term == 177:
                        term_name_wm = "คิมม"
                    elif num_term == 288:
                        term_name_wm = "มนน"
                    elif num_term == 399:
                        term_name_wm = "แซมม"
                    # append water mask
                    term_wm = {"term":term_name_wm,
                        "score":score_wm}
                    term_list.append(term_wm)
                # delete last three words
                if num_term not in [998,999,1000]:
                    term_list.append(term)
                # ----- end water mask -----
            # add topic-term to list
            topic_term = {"topic_no":x,
                        "terms":term_list}
            topic_term_dist.append(topic_term)
            x+=1
            print('-'*20)
        # print(topic_term_dist)
        return topic_term_dist  

    @staticmethod
    def document_dist(doc_id, title, text, id_ ,dictionary2,ldamodel,doc_topic_dist):
        bow = dictionary2.doc2bow(text)
        doc_dist = ldamodel.get_document_topics(bow, minimum_probability=0, minimum_phi_value=None,per_word_topics=False)
        print(doc_id, title)

        doc_topic_list = []
        for i in doc_dist:
            doc_topic_dict = {'topic_no':i[0]+1,'score':i[1]}
            doc_topic_list.append(doc_topic_dict)
        doc_dict = {'doc_id':doc_id,'topics':doc_topic_list}
        doc_topic_dist.append(doc_dict)


        print(doc_dist)
        print('-------------------------------------')
        return doc_topic_dist

    @staticmethod
    def docTopic_dist(doc_topic_dist,data_df, num_doc, inp_list,dictionary2,ldamodel):
        """
        Document-Topic Distribution is a probability score of document in topic.
        This method is called document_dist() which 

        Parameters
        ----------
        ldamodel: a topic model, required.
        corpus: a copus list of multiplied frequency of document's title with defined number, required.
            
        Returns
        ----------
        topic_term_dist: a list of dictionaries which contain topics and 1000 terms with their score in lambda (λ) = 0.6
            [
                {
                    'topic_id': 0, 
                    'terms': [
                        {
                            'term': 'ทุจริต', 
                            'score': 203.54537082463503
                        },...
                    ]
                },...
            ]
        """
        for i in range(num_doc):
            doc_id = data_df['doc_id'][i]
            title = data_df['title'][i]
            content = inp_list[i]
            doc_topic_dist = TextDistribution.document_dist(doc_id, title, content, i,dictionary2,ldamodel,doc_topic_dist)
        return doc_topic_dist

    @staticmethod
    def document_dist_min(doc_id, title, text, doc_dist_dict, dictionary2, ldamodel):
        bow = dictionary2.doc2bow(text)
        doc_dist = ldamodel.get_document_topics(bow, minimum_probability=0.1, minimum_phi_value=None,per_word_topics=False)
        print(doc_id, title)

       # add to list
        for i in doc_dist:
            # print(doc_dist_dict)
            # print(i[0])
            # print(doc_dist_dict)
            if i[0] in doc_dist_dict:
                doc_dist_dict[i[0]] += 1
            # print(i, end=' ')
            # print(doc_dist_dict)

        print()
        print('-------------------------------------')
        return doc_dist_dict

    @staticmethod
    def Ndoc_topic(n_doc_intopic,num_doc, data_df, inp_list, dictionary2, ldamodel):

        doc_dist_dict = {}
        for i in range(num_doc):
            doc_dist_dict[i] = 0
        print(doc_dist_dict)
        for i in range(num_doc):
            doc_id = data_df['doc_id'][i]
            title = data_df['title'][i]
            content = inp_list[i]
            doc_dist_dict = TextDistribution.document_dist_min(doc_id, title, content,doc_dist_dict, dictionary2, ldamodel)

        print(doc_dist_dict)
        for i in doc_dist_dict:
            ndoc_dict = {'topic_no':i+1, 'n_doc':doc_dist_dict[i]}
            n_doc_intopic.append(ndoc_dict)

        return n_doc_intopic

    
    @staticmethod
    def compute_term_pairs(topic_term_dist, no_top_terms = 30):
        """
        This method computes the co-occurence of word pais across different topics. Each word in a pair must be from different topics.

        Param:

            "term_topic_matrix":[
                {
                    "topic_no":1, //from 1-10
                    "terms":[
                        {
                           "term":"prawns",
                           "score":1.0265928777676891
                        },
                        {
                           "term":"long",
                           "score":1.0263626183179442
                        },
                        {
                           "term":"Recruitment",
                           "score":1.0260714211530289
                        }
                    ]
                },
                {
                    "topic_no":2,
                    "terms":[
                        {
                           "term":"prawns",
                           "score":1.0265928777676891
                        },
                        {
                           "term":"long",
                           "score":1.0263626183179442
                        },
                        {
                           "term":"Recruitment",
                           "score":1.0260714211530289
                        }
                    ]
                }
           ]


        Return:

        Example 
            "term_pairs":[
                {
                    "term_1":"prawns",
                    "term_2":"hello",
                    “cooccur_score”:0.789
                },
                ...
            ]


        """

        term_pairs = []
        max_cooccurence_score = 0
        # print(topic_term_dist)
        # print("No of Topics: {0}".format(len(topic_term_dist)))
        for each_term_topic_1 in topic_term_dist:
            topic_no_1 = each_term_topic_1['topic_no']
            # print("topic id 1: {0}".format(topic_no_1))

            for each_term_topic_2 in topic_term_dist:
                topic_no_2 = each_term_topic_2['topic_no']

                # compare terms from two different topics as half (triangle) matrix regardless of the same number of topic_no
                if (int(topic_no_1) < int(topic_no_2)):
                    # print("topic id 2: {0}".format(topic_no_2))

                    terms_1 = each_term_topic_1['terms']
                    for i in range(min(len(terms_1), no_top_terms)):
                        term_pair = {}
                        term_1 = terms_1[i]
                        score_1 = term_1['score']
                        # print('topic id {0}, term {1}: "{2}": score={3}'.format(topic_no_1, i, term_1['term'], score_1))

                        terms_2 = each_term_topic_2['terms']
                        for j in range(min(len(terms_2), no_top_terms)):
                            term_2 = terms_2[j]
                            score_2 = term_2['score']
                            # print('topic id {0}, term {1}: "{2}": score={3}'.format(topic_no_2, j, term_2['term'], score_2))

                            if(term_1['term'] !=  term_2['term']):
                                cooccurence_score = score_1 * score_2
                                term_pair['term_1'] = term_1['term']
                                term_pair['term_2'] = term_2['term']
                                term_pair['cooccur_score'] = cooccurence_score
                                if max_cooccurence_score < cooccurence_score:
                                    max_cooccurence_score = cooccurence_score
                                # print('{0}, {1}: co-occurence = {2}'.format(term_1['term'], term_2['term'], cooccurence_score))
                                # print("------------------------------------------------------------------------------------")

                                term_pairs.append(term_pair)

                            term_pair = {}

        # print("============before sorting===========")
        # print(term_pairs)

        # print("+++++++++++++after sorting+++++++++++++")
        sorted_term_pairs = sorted(term_pairs, key=lambda i: i['cooccur_score'], reverse=True)
        # print(sorted_term_pairs)

        counter = 0
        for term_pair in sorted_term_pairs:
            term_pair['cooccur_percent'] = (term_pair['cooccur_score']/max_cooccurence_score)*100
            sorted_term_pairs[counter] = term_pair
            counter += 1

        return sorted_term_pairs