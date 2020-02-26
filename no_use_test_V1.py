# pdfreader
from pylexto import LexTo
from PDFreader.pdfReader import extract_pdf

#docx2xt
import docx2txt

from split_word import split_word 
import matplotlib.pyplot as plt
from no_use_clean_doc import clean_alphabet

from gensim import corpora, models
import numpy as np
import pandas as pd
from numpy import percentile
from no_use_outlier import removing_outlier
from no_use_zipf_law import zipf_law

print("========== PART 1 : Input Dataset ==========")
data_file = []
data_file.append('FinalReport_Sample/pdf/RDG56A030_full.pdf') 
data_file.append('FinalReport_Sample/pdf/RDG60T0048V01_full.pdf') 

# Checking format file
for i in range(len(data_file)):
    if data_file[i].endswith('.pdf'):
        print("Document",i+1, " is pdf file.")
        print("filename:", data_file[i])
        # pdf document
        data_file[i] = extract_pdf(data_file[i])
    elif data_file[i].endswith('.docx'):
        print("Document",i+1, " is docx file.")
        # docx document
        data_file[i] = docx2txt.process(data_file[i])

print("========== PART 2 : Cleaning Data and Creating Word Tokenize ==========")
data_ready = []
for i in range(len(data_file)):
    print("------- Document",i+1,"-----------")
    # Cleaning document which have special alphabet 
    cleaned_data = clean_alphabet(data_file[i])
    # Split word in document
    data_ready.append(split_word(cleaned_data))
    # print("-------------------------")

print("========== PART 3 : Bag of word ==========")
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(data_ready)
# dict2 = {dictionary[ID]:ID for ID in dictionary.keys()}

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in data_ready]
# print("Corpus: ",corpus)

counts = {}
# process filtered tokens for each document
for doc_tokens in data_ready:
    for token in doc_tokens:
        # increment existing?
        if token in counts:
            counts[token] += 1
        # a new term?
        else:
            counts[token] = 1
# print("Found %d unique terms in this corpus" % len(counts))
# print(counts)

# sort frequency count of unique word
import operator
sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
# print(sorted_counts)

print("========== PART 3 : Outlier or Zipf's Law ==========")
# ------ Detecting and Removing Outlier ------
# corpus_remove_outlier = removing_outlier(sorted_counts, dictionary)
# print(corpus_remove_outlier)

# ------ Zipf's Law -------
count_vector = zipf_law(sorted_counts, dictionary)