import re
import string
from pythainlp.word_vector import sentence_vectorizer 

def clean_alphabet(document):
    # ลบวงเล็บ
    document = re.sub(r'\(','', document)
    document = re.sub(r'\)','', document)
    document = re.sub(r'\[\n','', document)
    document = re.sub(r'\]\n','', document)
    document = re.sub(r'±','', document)
    
    # ลบ hashtag
    document = re.sub(r'\n','',document)
    
    #ลบ dash
    document = re.sub(r'-','',document)
    document = re.sub(r'–','', document)
    
    #ลบอักขระพิเศษ
    document = re.sub(r'\x00','', document)
    document = re.sub(r'\uf03d','', document)
    document = re.sub(r'\uf050','', document)
    document = re.sub(r'\uf0b4','', document)
    document = re.sub(r'\uf044','', document)
    document = re.sub(r'§','',document)
    document = re.sub(r'■','',document)
    document = re.sub(r'”','',document)
    document = re.sub(r'“','',document)
    document = re.sub(r'‘','',document)
    document = re.sub(r'’','',document)
    document = re.sub(r'','',document)
    document = re.sub(r'•','',document)
    document = re.sub(r'™','',document)
    document = re.sub(r'≤','',document)
    document = re.sub(r'≥','',document)
    document = re.sub(r'……','',document)
    document = re.sub(r'…','',document)
    
    
    
    
    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        document = re.sub(r'\{}'.format(c),' ',document)
    
    # ลบ separator เช่น \n \t
    document = ' '.join(document.split())
    
    return document

def clean_thaistopwords():
    thaistopwords = []
    with open('corpus/stopwords-th.txt', 'r', encoding='utf-8-sig') as my_file:
        for line in my_file:
            cleanthaistopword = re.sub("\n", "", line)
            thaistopwords.append(cleanthaistopword)
    # print(thaistopwords)
    return thaistopwords