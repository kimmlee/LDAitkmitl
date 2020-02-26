import numpy as np
import pandas as pd
def zipf_law(sorted_counts, dictionary):
    wordid = []
    word_origin = []
    for ID in dictionary.keys():
        wordid.append(ID)
        # print(ID, dictionary[ID])
        word_origin.append(dictionary[ID])

    word_ranking = []
    word_list = []
    count_list = []
    x = 1
    for word, count in sorted_counts:
        word_ranking.append(x)
        word_list.append(word)
        count_list.append(count)
        x+=1

    sum_wordCount = sum(count_list)
    # print("Sum of word count: ",sum_wordCount)

    df= pd.DataFrame({"wordToken":word_list, "wordCount":count_list,"r":word_ranking}) 
    df['Pr'] = df['wordCount']/sum_wordCount
    df['C'] = df['r'] * df['Pr']
    # print(df.head(20))
    print(df.describe())
    print(df.head(20))

    # ------- Plot Graph for removing data ------- 
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # df_graph=pd.DataFrame({'xvalues': df['r'], 'yvalues': df['C'] })
    # plt.plot( 'xvalues', 'yvalues', data=df_graph)
    # plt.show()

    # Removing frequency data
    df1=df
    df1.drop(df1[ df1['C'] < 0.15 ].index, inplace=True)
    # print(df1)
    print(df1.describe())
    print(df.head(20))

    # Bring list of words and count of words back to corpus type
    counts = []
    words = df1['wordToken'].values.tolist()
    words_id = []
    for num in range(0, len(word_origin)):
        for word in words:
            if word == word_origin[num]:
                # print("Origin word: ",wordid[num], word_origin[num])
                words_id.append(wordid[num])
                counts.append(df1['wordCount'].loc[df1['wordToken'] == word_origin[num]].values[0])
    zipbWord = zip(words_id, counts)
    wordlist = list(zipbWord)
    corpus_remove_outlier = [wordlist]
    print(corpus_remove_outlier)
    return corpus_remove_outlier










