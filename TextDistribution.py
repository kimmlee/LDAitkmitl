
class TextDistribution:

    @staticmethod
    def topicTerm_dist(dic,corpus,topic_term_dist, term_dist_topic):
        #lambda = 1.0
        term_dist_topic = dict(term_dist_topic)
        for x in term_dist_topic:
            term_list = []
            for num_word in range(len(term_dist_topic[0])):
                term_name = term_dist_topic[x][num_word][0]
                score = term_dist_topic[x][num_word][1]
                term = {"term":term_name,
                        "score":score}
                # ----- start water mask -----
                if num_word in [177,288,399]:
                    score_next = term_dist_topic[x][num_word+1][1]
                    score_wm = ((score - score_next)/2)+score_next
                    if num_word == 177:
                        term_name_wm = "คิมม"
                    elif num_word == 288:
                        term_name_wm = "มนน"
                    elif num_word == 399:
                        term_name_wm = "แซมม"
                    # append water mask
                    term_wm = {"term":term_name_wm,
                        "score":score_wm}
                    term_list.append(term_wm)
                # delete last three words
                if num_word not in [998,999,1000]:
                    term_list.append(term)
                # ----- end water mask -----
            # add topic-term to list
            topic_term = {"topic_id":x,
                        "terms":term_list}
            topic_term_dist.append(topic_term)
        
        #lambda = 0.6
        _lambda = 0.6
        topic_term_dist_6 = []
        for topic in topic_term_dist:
            term_list = topic['terms']
            new_term_list = []
            # print(term_list)
            for term in term_list:
                term_name = term['term']
                term_score = term['score']
                if term_name in ['คิมม',"มนน","แซมม"]:
                    new_term_score = term_score
                else:
                    term_id = dic[term_name]
                    prob_w = TextDistribution.prob_word(corpus, term_id)
                    prob_w_t = term_score
                    new_term_score = (_lambda * prob_w_t) + ((1 - _lambda)*(prob_w_t/prob_w))
                new_term = {"term":term_name,
                        "score":new_term_score}
                new_term_list.append(new_term)
            topic_term = {"topic_id":topic['topic_id'],
                        "terms":new_term_list}
            topic_term_dist_6.append(topic_term)
            
        return topic_term_dist_6

    @staticmethod
    def prob_word(corpus, term_id):
        corpus_list = []
        for cp in corpus:
            for word in cp:
                corpus_list.append(word)
        sum_ = sum(dict(corpus_list).values())
        tfi = dict(corpus_list)[term_id]
        # print(tfi, sum_)
        prob_w = tfi/sum_
        # print(prob_w)
        return prob_w

    @staticmethod
    def document_dist(doc_id, title, text, id_ ,dictionary2,ldamodel,doc_topic_dist):
        bow = dictionary2.doc2bow(text)
        doc_dist = ldamodel.get_document_topics(bow, minimum_probability=0, minimum_phi_value=None,per_word_topics=False)
        print(doc_id, title)

        doc_topic_list = []
        for i in doc_dist:
            doc_topic_dict = {'topic_id':i[0],'score':i[1]}
            doc_topic_list.append(doc_topic_dict)
        doc_dict = {'doc_id':doc_id,'topics':doc_topic_list}
        doc_topic_dist.append(doc_dict)


        print(doc_dist)
        print('-------------------------------------')
        return doc_topic_dist

    @staticmethod
    def docTopic_dist(doc_topic_dist,data_df, num_doc, inp_list,dictionary2,ldamodel):

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
                print("Here")
                doc_dist_dict[i[0]] += 1
            print(i, end=' ')
            print(doc_dist_dict)

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
            ndoc_dict = {'topic_id':i, 'n_doc':doc_dist_dict[i]}
            n_doc_intopic.append(ndoc_dict)

        return n_doc_intopic