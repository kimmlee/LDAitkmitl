import numpy as np
import pandas as pd
from numpy import percentile
def removing_outlier(sorted_counts, dictionary):

    wordid = []
    word_origin = []
    for ID in dictionary.keys():
        wordid.append(ID)
        # print(ID, dictionary[ID])
        word_origin.append(dictionary[ID])

    word_list = []
    count_list = []
    for word, count in sorted_counts:
        word_list.append(word)
        count_list.append(count)

    print("========== PART 3.1 : Detecting Outlier ==========")
    # list to dataframe
    df= pd.DataFrame({"wordToken":word_list, "wordCount":count_list}) 
    # print(df) 

    # --------- Identify outliers with interquartile range (IQR) ----------
    data = df['wordCount']
    print(df.describe())

    # Calculate interquartile range
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25
    # print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

    # Calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    # Identify outliers
    outliers = [x for x in data if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))

    print("========== PART 3.2 : Removing Outlier ==========")
    # Remove outliers
    outliers_removed = [x for x in data if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    print(outliers)
    outlier_u = np.unique(np.array(outliers))
    df1 = df
    for i in outlier_u:
        df1['wordCount'] = df1['wordCount'].replace(to_replace=i, value=np.nan) #Replace with Nan and will drop this column
    df1 = df1.dropna()
    print(df1)
    print(df1.describe())
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
    return corpus_remove_outlier