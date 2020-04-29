from pythainlp.tokenize import word_tokenize
import pythainlp.corpus
from pythainlp.corpus import thai_words, thai_stopwords
from pythainlp.tag import pos_tag
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import string
import pandas
import re
"""
    A TextPreProcessing class is a preprocessing text in document file of methods to handles spliting words to create word token, removing unwanted words and 
    increasing frequency of word before putting it to LDAmodel. 
    This class contains 4 static methods, including: 
        1) split_word()
        2) cut_character()
        3) postag()
        4) add_frequency()
"""
class TextPreProcessing:
    

    @staticmethod
    def split_word(text):
        """
        A static method, split_word(text), splits word by newmm word tokenize which method of pythainlp, cleans special character and removes stop word
        
        Parameters
        ----------
        text: a content of each document file in a string format, required
            
        Returns
        ----------
        tokens: a list contains preprocessing string word from a document text (raw text.)
            ['บ้าน','เปร็ด','โรงพยาบาล',...]
        """
        train_text = ""
        #Remove special charecter
        pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")
        remove_char = re.findall(pattern, text)
        list_with_removed_char = [char for char in text if not char in remove_char]
        train_text = ''.join(list_with_removed_char)

        #Split word by using for Maximum Matching algorithm,
        tokens = word_tokenize(train_text, engine='newmm')
        #Remove thai stop word and eng stop word
        stopped_tokens = [i for i in tokens if not i in thai_stopwords() and i not in get_stop_words('en')]
        #Word stemming
    #     stemmed_tokens = [PorterStemmer().stem(i) for i in stopped_tokens]
        #Remove ETC
        deletelist = [' ','  ','   ', '    ','     ','      ','       ','                                ', \
            '        ','\n', '\xa0','\x0c', "'",'cid']
        tokens = [i for i in stopped_tokens if not i in deletelist]

        return tokens

    @staticmethod
    def cut_character(inp_list, shortest_num_cut, longest_num_cut):
        """
        A static method, cut_character(inp_list, num_cut), removes word token which is equal or fewer than defined the number of characters.
        
        Parameters
        ----------
        inp_list: a list of all document's token list, just like
        Example.
        [
            ['สน','สายหยุด','ดอกไม้',...],
            ['คน','ประเทศไทย','ใส่ใจ',...],
            ['จน','พัฒนา','ศักยภาพ',...]
        ]
        num_cut: the number of character that you want to cut it off from token
        Example.
         if num_cut is 2, word like คน, สน, จน will be removed
            
        Returns
        ----------
        new_lists: a new list of inp_list which removed word that equal or fewer than defined number of character.
        [
            ['สายหยุด','ดอกไม้',...],
            ['ประเทศไทย','ใส่ใจ',...],
            ['พัฒนา','ศักยภาพ',...]
        ]
        """

        count = 0
        for i in range(len(inp_list)):
            for j in range(len(inp_list[i])):
                if (len(inp_list[i][j]) <= shortest_num_cut) or (len(inp_list[i][j]) > longest_num_cut):
                    count += 1
                    inp_list[i][j] = '@@@@@@@@@'

        print("The number of short and long words removed is {0} tokens.".format(count))
        count = 0
        for i in inp_list:
            count += len(i)
        print(count)

        new_lists = []
        for i in range(len(inp_list)):
            temp = []
            for j in inp_list[i]:
                if j != '@@@@@@@@@':
                    temp.append(j)
            new_lists.append(temp)

        count = 0
        for i in new_lists:
            count += len(i)
        print(count)
        return new_lists

    @staticmethod
    def postag(words_list):
        """
        A static method, postag(words_list), removes word is not noun and prop noun by Pythainlp's pos_tag function.
        
        Parameters
        ----------
        words_list: a list of all document's token list.
            
        Returns
        ----------
        pos_list: a new list of words_list which removed word is not noun and prop noun.
        [
            ['สายหยุด','ดอกไม้',...],
            ['ประเทศไทย','ใส่ใจ',...],
            ['พัฒนา','ศักยภาพ',...]
        ]
        """

        word_with_pos = pos_tag(words_list)
        pos_list = []
        for word in word_with_pos:
            if word[1] in ['NOUN','NCMN','NTTL','CNIT','CLTV','CMTR','CFQC','CVBL','PROPN','NPRP']:
                pos_list.append(word[0])
        return pos_list

    @staticmethod
    def add_frequency(dict_2, corpus, data_df, freq_multiply, num_doc):
        """
        A static method, postag(words_list), increases frequency of words which are title of document.
        
        Parameters
        ----------
        dict_2: a dictionary of word token (key) and word token id (value) which collected all document's token list.
        corpus: list of all document's frequency word token tuple in dictionary which contains word_token_id and frequency_of_word_token.
        data_df: DataFrame is a 2-dimensional labeled data structure with columns of potentially different types 
                 which contains document id (doc_id), title and content of each document.
        freq_multiply: the number multiplied by the frequency.
        num_doc: the number of documents input.
            
        Returns
        ----------
        corpus: a new corpus that multiplied frequency of document's title with defined number.
        [
            [
                (0, 1),
                (1, 2),
                (2, 1),
                (3, 1),
                ...
            ]
        ]
        """

        title_list = []
        for num in range(num_doc):
            title = data_df['title'][num]
            title_list.append(TextPreProcessing.split_word(title))

        print(title_list)

        # Add title of dict_2 index in index list
        title_index = []
        for num in range(num_doc):
            index_list = []
            for j in range(len(title_list[num])):
                if title_list[num][j] in dict_2:
                    if dict_2[title_list[num][j]] not in index_list:
                        index_list.append(dict_2[title_list[num][j]])
        #                 print(header_list[i][j], dict_2[header_list[i][j]])
            title_index.append(index_list)

        print(title_index)

        # Plus frequency in corpus tuple by searching index of dict
        for i in range(num_doc):
            # tuple to dict
            dict_corpus = dict(corpus[i])
            dict_update = {}
            for j in range(len(title_index[i])):
                if title_index[i][j] in dict_corpus.keys():
                    # get frequecy of index, plus, and update it
                    frequency = dict_corpus[title_index[i][j]]
            #             print(index_list[j], frequency*freq_multiply)
                    dict_update[title_index[i][j]] = frequency*freq_multiply

            # update frequency of title
            dict_corpus.update(dict_update)
            # dict to tuple
            corpus[i] = list(dict_corpus.items())

        return corpus