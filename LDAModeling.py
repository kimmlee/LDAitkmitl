import sys

sys.path.append("..")  # Adds higher directory to python modules path.

from Util import Util
from TextPreProcessing import TextPreProcessing
from TextDistribution import TextDistribution

import pyLDAvis.gensim
from gensim import corpora, models
from gensim.models import LsiModel, LdaModel, CoherenceModel

import pandas as pd

import bs4
from service_helper import send_progress
from service_helper import filename_from_request

class LDAModeling:
    """
    LDAModeling class is The process of Latent Dirichlet Allocation (LDA), a form of unsupervised learning which generate a topic modeling is 
    the process of identifying topics in a set of documents. This can be useful for search engines, customer service 
    automation, and any other instance where knowing the topics of documents is important.
    This class contains 4 static methods, including: 
        1) to_dataframe()
        2) localize_pyLDAvis_to_thai()
        3) LDAModel()
        4) perform_topic_modeling()

    Attributes
    ----------
    shortest_num_cut: integer
        The number of word character is less than input number will be remove.

    longest_num_cut: integer
        The number of word character is longer than input number will be remove.

    no_top_terms: integer, optional (default = 20) set inside TextDistribution.compute_term_pairs()
        The maximum number of term of topic that will be pair

    max_returned_term_pairs: integer, optional (default = -1) -1 mean no limit of term pairs
        The maximum number of term of topic that will be pair

    """

    def __init__(self):
        self.shortest_num_cut = 2
        self.longest_num_cut = 100
        self.no_top_terms = 20
        self.max_returned_term_pairs = 100
        self.sort_topics = False

    def to_dataframe(self, data, titles, doc_path_file):
        """
        A static method, change a document in dictionary to dataframe and setting field like...
        | doc_id | title | content |

        Parameters
        ----------
        doc_id: string, a document's file name.
        title: string, a title of document.
        content: string, a content of document.
            
        Returns
        ----------
        data_df: a data frame, in which keys are document id, title, and content.
        | doc_id | title  |  content   |
        |  001   | การศึกษาวิเคราะห์การทุจริตคอร์รัปชันของขบวนการเครือข่ายนายหน้าข้ามชาติในอุตสาหกรรมประมงต่อเนื่องของประเทศไทย | โครงการวิจัยและพัฒนาแนวทางการหนุนเสริมทางวิชาการเพื่อพัฒนากระบวนการผลิตและพัฒนาครูโดยบูรณาการแนวคิดจิตตปัญญาศึกษา |

        """
        data_doc = []
        data_titles = titles
        data_content = []
        for doc_name in data.keys():
            data_content.append(data[doc_name][0])
            for key,value in doc_path_file.items():
                if doc_name in value:
                    id_ = key
            data_doc.append(id_)
        data_df_dict = {'doc_id': data_doc, 'title': data_titles, 'content': data_content}
        data_df = pd.DataFrame.from_dict(data_df_dict)
        return data_df

    def get_unreadable_doc_ids(self, unreadable_doc_names, doc_path_file):
        unreadable_docs = []
        for unreadable_doc_name in unreadable_doc_names:
            for key, value in doc_path_file.items():
                if unreadable_doc_name in value:
                    id_ = key
            unreadable_docs.append(id_)
        return unreadable_docs


    def localize_pyLDAvis_to_thai(self, project_name, en_input_dir, en_pyLDAvis_file, th_output_dir, th_pyLDAvis_file):
        """
        A static method, change a original pyLDAvis html to thai version

        Parameters
        ----------
        project_name: string
            Input project name
        en_input_dir: string
            English input path directory 
        en_pyLDAvis_file: string
            English pyLDAvis html file name
        th_output_dir: string
            Thai input path directory 
        th_pyLDAvis_file: string
            Thai pyLDAvis html file name

        Returns
        ----------
        pyLDAvis version Thai in th_output_dir path
        """
        with open(en_input_dir + en_pyLDAvis_file) as inf:
            txt = inf.read()
            soup = bs4.BeautifulSoup(txt, features="lxml")

        meta = soup.new_tag("meta", charset="utf-8")
        soup.head.append(meta)

        new_title_tag = soup.new_tag("title")
        new_title_tag.string = project_name
        soup.head.append(new_title_tag)

        souptemp = soup.prettify()
        souptemp = souptemp.replace('https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js', '/static/js/d3.min.js')
        souptemp = souptemp.replace('https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js', '/static/js/ldavis.v1.0.0.js')
        souptemp = souptemp.replace('https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css', '/static/css/ldavis.v1.0.0.css')

        with open(th_output_dir + th_pyLDAvis_file, "w") as outf:
            outf.write(souptemp)

    """to remove"""

    # Generate LDA Model
    def LDAmodel(self, dictionary, corpus, num_top=10):
        """
        To generate LDA Model with gensim module

        Parameters
        ----------
        dictionary: dictionary
            A dictionary of word and word id in document
        corpus: list
            A list of word id and its frequency in document
        num_top: integer, optional (default = 10)
            The number of topic to define 

        Returns
        ----------
        LDAmodel object
        """
        ldamodel = LdaModel(corpus, num_top, id2word=dictionary, decay=0.6, random_state=2, passes=10)
        return ldamodel

    def perform_topic_modeling(
        self, 
        id, 
        project_id, 
        project_name, 
        input_local_root, 
        files, 
        titles,
        converted_local_root,
        doc_path_file,
        output_dir, 
        pyLDAvis_output_file, 
        th_output_dir, 
        th_pyLDAvis_output_file,
        undownloadable_documents,
        max_no_topic=10, 
        are_short_and_long_words_removed=True):

        """
        The process of building topic modeling and generating topic-term distribution, which have 8 steps:
            1) Filter input file to read (pdf/docx to text)
            2) Data preparation and creating word tokenization
            3) Generate LDA Model
            4) Topic Term-distribution
                4.1) Document-topic (all) distribution 
                4.2) Document-topic (min) distribution
            5) Evaluate Model (optional)
            6) Export pyLDAvis HTML 
            7) Convert pyLDAvis HTML to Thai
            8) Word/Term Pair Similairty

        Parameters
        ----------
        project_name: string
        The name of project which received from json input data.

        input_local_root: string
            A string of downloaded file path save to local root. 

        files: list of filename
            A list of filename that completed download to local path.

        titles: list of title file
            A list of title completed download file

        doc_path_file: dictionary of path file
            A dictionary of local file path, key is document id (doc_id) and value is local file path.

        converted_local_root: string
            A converted file directory to save an converted input file from unreadable.

        output_dir: string
            An output 'directory' to save an original pyLDAvis html file.

        pyLDAvis_output_file: string
            An output 'file name' to save an original pyLDAvis html file.

        th_output_dir: string
            An output 'directory' to save an thai pyLDAvis html file.

        th_pyLDAvis_output_file: string
            An output 'file name' to save an thai pyLDAvis html file.
        
        max_no_topic: integer, optional (default = 10)
            A maximum number of topic requested input from user.
        
        are_short_and_long_words_removed: boolean, optional (default = True)
            A boolean flag variable to set remove character number function.

        Returns
        ----------
        result : a dictionary in which key are the result of performing topic modeling.
            {
                "project_id": null,
                "success": True/False,
                "errorMessage": null,
                "topic_chart_url": output_dir + pyLDAvis_output_file,
                "term_topic_matrix": topic_term_dist,
                "document_topic_matrix":doc_topic_dist,
                "topic_stat":n_doc_intopic,
                "term_pairs":terms_pairs,
                "unreadable_documents":["..","..",..]
            }
        """

        error_payload = {
            "unreadable_documents": [],
            "undownloadable_documents": undownloadable_documents
        }

        if len(files) == 0 and len(titles) == 0:
            print("[E] Both the number of input files and the number of titles are zero. Canceling this job...")
            send_progress(id=id, code="601", keep=True, data=str(error_payload))
            return
        elif len(files) != len(titles):
            print("[E] The number of input files is not equal to the number of titles. Canceling this job...")
            send_progress(id=id, code="602", keep=True, data=str(error_payload))
            return

        print("========== PART 1 : Input Files ==========")
        send_progress(id=id, code="110", keep=True)
        data, unreadable_doc_names = Util.filter_file_to_read(id, input_local_root, files, converted_local_root)
        num_doc = len(titles)

        if len(data) != num_doc:
            if len(unreadable_doc_names) > 0:
                error_payload['unreadable_documents'] = self.get_unreadable_doc_ids(unreadable_doc_names, doc_path_file)
            send_progress(id=id, code="603", keep=True, data=str(error_payload))
            return

        send_progress(id=id, code="120", keep=True)


        print("========== PART 2 : Data Preparation and Creating Word Tokenization ==========")
        # Set data into dataframe type
        data_df = self.to_dataframe(data, titles, doc_path_file)
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
        dict2 = {dictionary[ID]: ID for ID in dictionary.keys()}

        # Convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in inp_list]
        tfidf = models.TfidfModel(corpus, smartirs='ntc')
        corpus_tfidf = tfidf[corpus]

        if are_short_and_long_words_removed:
            # Remove character number is less than 2 words off
            new_lists = TextPreProcessing.cut_character(inp_list, self.shortest_num_cut, self.longest_num_cut)
        else:
            new_lists = inp_list

        # Remove word is not noun and prop noun by pos_tag function
        for num in range(num_doc):
            send_progress(id=id, code="121", payload=[num + 1])
            new_lists[num] = TextPreProcessing.postag(new_lists[num])

        # Create new dict and corpus
        dictionary2 = corpora.Dictionary(new_lists)
        dict_2 = {dictionary2[ID]: ID for ID in dictionary2.keys()}
        corpus2 = [dictionary2.doc2bow(text) for text in new_lists]

        # Header Title plus frequency in corpus
        corpus2 = TextPreProcessing.add_frequency(dict_2, corpus2, data_df, 10, num_doc)

        print("========== PART 3 : Generate LDA Model ==========")
        # Generate LDA Model

        send_progress(id=id, code="140", keep=True)

        # Default number of topic is 10. If the number of documents is fewer than the maximum number of topics, the number of documents will be used to as the maximum number of topics.
        max_no_topic = min([max_no_topic, num_doc])
        if max_no_topic < 2:
            max_no_topic = 2

        ldamodel = self.LDAmodel(dictionary2, corpus2, max_no_topic)
        term_dist_topic = ldamodel.show_topics(num_topics=max_no_topic, num_words=1000, log=True, formatted=False)
        # print(term_dist_topic)

        print("========== PART 4 : Topic-term distribution ==========")
        ### Topic-Term Dist
        topic_term_dist = []
        # print(dictionary2['ทุจริต'])
        # print(dictionary2)
        topic_term_dist = TextDistribution.topicTerm_dist(ldamodel, corpus2, sort_topics=self.sort_topics)
        # print(topic_term_dist)

        print("========== PART 4-1 : Document-topic (all) distribution ==========")
        ### Doc_topic_all_dist
        doc_topic_dist = TextDistribution.docTopic_dist(data_df, num_doc, inp_list, dictionary2, ldamodel)
        # print(doc_topic_dist)

        print("========== PART 4-2 : Document-topic (min) distribution ==========")
        ### Doc_topic_min_dist
        n_doc_in_topic = TextDistribution.num_doc_topic(num_doc, data_df, inp_list, dictionary2, ldamodel, max_no_topic)
        # print(n_doc_intopic)

        print("========== PART 5 : Evaluate Model ==========")

        send_progress(id=id, code="160", keep=True)

        # Evaluate
        # lda_coherence = CoherenceModel(ldamodel, corpus=corpus2, dictionary=dictionary2, coherence='u_mass')
        # print(lda_coherence.get_coherence_per_topic())
        # print("LDA umass score = %.4f" % (lda_coherence.get_coherence()))
        # lda_coherence = CoherenceModel(ldamodel, texts=new_lists, dictionary=dictionary2, coherence='c_uci')
        # print("LDA uci score = %.4f" % (lda_coherence.get_coherence()))

        print("========== PART 6 : Export pyLDAvis HTML ==========")

        send_progress(id=id, code="170", keep=True)

        # pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(ldamodel, corpus2, dictionary=ldamodel.id2word, sort_topics=self.sort_topics)
        pyLDAvis.save_html(vis, output_dir + pyLDAvis_output_file)

        print("========== PART 7 : Convert pyLDAvis HTML to Thai==========")
        self.localize_pyLDAvis_to_thai(project_name, output_dir, pyLDAvis_output_file, th_output_dir,
                                       th_pyLDAvis_output_file)

        print("========== PART 8 : Word/Term Pair Similairty==========")
        terms_pairs = TextDistribution.compute_term_pairs_exp_max(topic_term_dist, self.no_top_terms, self.max_returned_term_pairs)
        # print(terms_pairs)

        send_progress(id=id, code="180", keep=True)
        topic_chart_url = filename_from_request(project_id)

        result = {
            "project_id": project_id,
            "topic_chart_url": topic_chart_url,
            "term_topic_matrix": topic_term_dist,
            "document_topic_matrix": doc_topic_dist,
            "topic_stat": n_doc_in_topic,
            "term_pairs": terms_pairs,
            "unreadable_documents": error_payload['unreadable_documents'],
            "undownloadable_documents": undownloadable_documents
        }

        output_files = [
            ('resultFile',
             (topic_chart_url['en'], open('./results/' + pyLDAvis_output_file, 'rb'), 'text/html', {'Expires': '0'})),
            ('resultFile',
             (topic_chart_url['th'], open('./results/' + th_pyLDAvis_output_file, 'rb'), 'text/html', {'Expires': '0'}))
        ]

        # send_progress(id=id, code="190", data=result, keep=True, files=output_files)
        send_progress(id=id, code="191", files=output_files)
        send_progress(id=id, code="192", data=str(result))

        send_progress(id=id, code="050", keep=True)
