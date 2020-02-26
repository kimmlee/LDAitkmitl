import sys
sys.path.append("..") # Adds higher directory to python modules path.

from Util import Util
from TextPreProcessing import TextPreProcessing
from TextDistribution import TextDistribution

import pyLDAvis.gensim
from gensim import corpora, models
from gensim.models import LsiModel, LdaModel, CoherenceModel

import pandas as pd

import bs4

"""An ensemble classifier that uses heterogeneous models at the base layer and a aggregation model at the 
    aggregation layer. A k-fold cross validation is used to generate training data for the stack layer model.

    Parameters
    ----------
    n_classes: integer, required (especially when setting proba_2train_meta = True)   
        The number of target classes for prediction, used to construct an array to store the prbability output from base classifiers

    estimators: list of BaseEstimators, optional (default = 6 base estimators, including:  
            1) DecisionTreeClassifier with max_depth = 10,
            2) ExtraTreeClassifier,
            3) LogisticRegression,
            4) Support Vector Machine - NuSVC,
            5) Support Vector Machine - SVC,
            6) KNeighborsClassifier with n_neighbors = 5)
        A list of scikit-learn base estimators with fit() and predict() methods.

    cv_folds: integer, optional (default = 5)
        The number of k to perform k-fold cross validation to be used. 

    proba_2train_meta: boolean, optional (default = False)
        A boolean flag variable to specify the meta learner at the stacked level should be trained either on label output or probability output.
            False => trained on 'label output' - Typical Super learner classifier, using Stack Layer Training Set (Labels)
            True => train on 'probability output' - using Stack Layer Training Set (Probabilities)

    ori_input_2train_meta: boolean , optional (default = False)
        A boolean flag variable to specify whether the meta learner should be trained on the training set with original features or not.
            False => trained only on 'output' from base estimators 
            True => train on 'original features/inputs' plus 'output' from base estimators

    meta_model_type: string, optional (default = "DCT")
        An option in string, only either "DCT" or "LR", to specify the type of model to use at the stack layer for a meta learner
            "DCT" => DecisionTreeClassifier
            "LR" => LogisticRegression

    est_accuracy: boolean, optional (default = False) - only available when proba_2train_meta = False
        A boolean flag variable to analyse and print the Mean Accuracy (MA) as the strength/performance of base estimators (predictive power). 
            False => do nothing
            True => analyse and print the mean accuracy

    est_corr: boolean, optional (default = False) - only available when proba_2train_meta = False
        A boolean flag variable to analyse and print the Pearson correlation between base estimators (diversity).
            False => do nothing
            True => analyse and print the Pearson correlation between base estimators

    debug_mode: boolean, optional (default = False)
        A boolean flag variable to set the verbose mode in printing debugging info via the "_print_debug" function

    Attributes
    ----------
    n_classes: integer   
        The number of target classes for prediction, used to expand an array to store the prbability output from base classifiers

    estimators: list
        A list of BaseEstimators used  

    cv_folds: integer
        the number of k to perform cross validation used 

    proba_2train_meta: boolean
        the flag variable to select the Stack Layer Training Set to be Labels (False) or Probabilities (True)

    ori_input_2train_meta: boolean
        the flag variable to include the original features/inputs in the Stack Layer Training Set: not include (False) or include (True)

    meta_model_type: string
        the option variable for the type of model of the meta-learner at the stack layer 

    est_accuracy: boolean
        the flag variable to perform and print an analysis of Mean Accuracy (MA) of base estimators: not perform (False) or perform (True)

    est_corr: boolean
        the flag variable to perform and print an analysis of Pearson correlation between base estimators: not perform (False) or perform (True)

    debug_mode: boolean
        The flag variable to print debugging info via the "_print_debug" function

    Notes
    -----


    See also
    --------

    ----------
    .. [1]  van der Laan, M., Polley, E. & Hubbard, A. (2007). 
            Super Learner. Statistical Applications in Genetics 
            and Molecular Biology, 6(1) 
            doi:10.2202/1544-6115.1309
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = SuperLearnerClassifier()
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)

    """
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
        souptemp = souptemp.replace('https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js', 'static/js/d3.min.js')
        souptemp = souptemp.replace('https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js', 'static/js/ldavis.v1.0.0.js')
        souptemp = souptemp.replace('https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css',
                                    'static/css/ldavis.v1.0.0.css')

        with open(th_output_dir + th_pyLDAvis_file, "w") as outf:
            outf.write(souptemp)


    """to remove"""
    # Generate LDA Model
    def LDAmodel(self, dictionary, corpus, num_top=10):
        ldamodel = LdaModel(corpus, num_top, id2word=dictionary, decay=0.6, random_state=2, passes=10)
        return ldamodel

    def perform_topic_modeling(self, input_local_root, files, titles, converted_local_root,
                               output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file,
                               max_no_topic = 10, is_short_words_removed = True):

        print("========== PART 1 : Input Files ==========")
        data = Util.filter_file_to_read(input_local_root, files, converted_local_root)
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
        max_no_topic = min([max_no_topic, num_doc])
        if max_no_topic < 2:
            max_no_topic = 2

        ldamodel = self.LDAmodel(dictionary2, corpus2, max_no_topic)
        term_dist_topic = ldamodel.show_topics(max_no_topic, 1000, log=True, formatted=False)
        # print(term_dist_topic)
        handle1=open('term_dist_topic','a+')
        handle1.write(str(term_dist_topic))
        handle1.write("\n")
        handle1.close()

        print("========== PART 4 : Topic-term distribution ==========")
        ### Topic-Term Dist
        topic_term_dist = []
        # print(dictionary2['ทุจริต'])
        # print(dictionary2)
        topic_term_dist = TextDistribution.topicTerm_dist(dict_2,corpus2,topic_term_dist, term_dist_topic)
        print(topic_term_dist)
        handle1=open('topic_term_dist','a+')
        handle1.write(str(topic_term_dist))
        handle1.write("\n")
        handle1.close()
        print("========== PART 4-1 : Document-topic (all) distribution ==========")
        ### Doc_topic_all_dist
        doc_topic_dist = []
        doc_topic_dist = TextDistribution.docTopic_dist(doc_topic_dist, data_df, num_doc, inp_list,dictionary2,ldamodel)
        print(doc_topic_dist)

        print("========== PART 4-2 : Document-topic (min) distribution ==========")
        ### Doc_topic_min_dist
        n_doc_intopic = []
        n_doc_intopic = TextDistribution.Ndoc_topic(n_doc_intopic,num_doc, data_df, inp_list, dictionary2, ldamodel)
        print(n_doc_intopic)

        print("========== PART 5 : Evaluate Model ==========")
        # Evaluate
        lda_coherence = CoherenceModel(ldamodel, corpus=corpus2, dictionary=dictionary2, coherence='u_mass')
        print(lda_coherence.get_coherence_per_topic())
        print("LDA umass score = %.4f" % (lda_coherence.get_coherence()))

        lda_coherence = CoherenceModel(ldamodel, texts=new_lists, dictionary=dictionary2, coherence='c_uci')
        print("LDA uci score = %.4f" % (lda_coherence.get_coherence()))

        print("========== PART 6 : Export pyLDAvis HTML ==========")
        # pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(ldamodel, corpus2, dictionary=ldamodel.id2word)
        pyLDAvis.save_html(vis, output_dir + pyLDAvis_output_file)

        print("========== PART 7 : Convert pyLDAvis HTML to Thai==========")
        self.localize_pyLDAvis_to_thai(output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file)