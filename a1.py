import os
import sys
import math
import pandas as pd
from glob import glob
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def part1_load(folder1, folder2, n=100):
    folders = []
    files = []
    big_dict = {}
    allfiles = glob("{}/*.txt".format(folder1)) + glob("{}/*.txt".format(folder2))
    column_names = ['Folder', 'File']

    for file in sorted(allfiles):
        my_dict = {}

        if file not in files:
            if folder1 in file:
                files.append(file.replace(folder1 + '/', ''))
                folders.append(folder1)
            else:
                files.append(file.replace(folder2 + '/', ''))
                folders.append(folder2)

        with open(file, 'r') as thefile:
            for l in thefile:
                for word in word_tokenize(l):
                    if word.isalpha():
                        word = word.lower()
                        if word in my_dict.keys():
                            my_dict[word] += 1
                        else:
                            my_dict[
                                word] = 1  # dictionary that contains the word as key and the value as count/document
        if folder1 in file:
            big_dict[file] = my_dict
        else:
            big_dict[file] = my_dict

    df = pd.DataFrame.from_dict({i: big_dict[i]
                                 for i in big_dict.keys()},
                                orient='index')

    sum_column = df.sum(axis=0)
    dictionary = sum_column[1:].to_dict()

    d = {}
    for key, value in dictionary.items():
        if value > n:
            d[key] = value

    final_dict = {}
    for data_key in df.keys():
        for k, v in d.items():
            if data_key == k:
                final_dict[data_key] = list(df[data_key])

    df2 = pd.DataFrame(final_dict)
    df2.insert(0, column_names[0], pd.Series(folders).values)
    df2.insert(0, column_names[1], pd.Series(files).values)
    df2.fillna(0, inplace=True)

    return df2


my_df = part1_load('crude', 'grain')


def part2_vis(df, m):
    assert isinstance(df, pd.DataFrame)
    df2 = df.groupby('Folder').sum()
    df = df2.transpose()
    d = df.to_dict()

    for key, value in d.items():
        d[key] = {k: v for k, v in sorted(value.items(), key=lambda item: item[1], reverse=True)}

    df2 = pd.DataFrame(d)
    df2.sort_values(by='grain', ascending=False)

    crude = list(df2['crude'])
    grain = list(df2['grain'])
    index = list(d['grain'].keys())

    df3 = pd.DataFrame({'crude': crude[:m],
                        'grain': grain[:m]}, index=index[:m])
    ax = df3.plot.bar(rot=0)
    ax.legend(title='class name')

    return ax

def part3_tfidf(df):
    assert isinstance(df, pd.DataFrame)
    s = df.iloc[:].sum(axis=1)
    keys = df.keys()
    keys = keys[2:]
    folders = df['Folder']
    files = df['File']
    d = {}
    for i in range(len(keys) - 1):
        tf = []
        total = 0
        tf_idf = []
        my_dict = df[keys[i]].to_dict()
        small_dict = s.to_dict()
        for key, value in my_dict.items():
            if small_dict[key] != 0:
                new_val = value / small_dict[key]
                if new_val != 0:
                    total += 1
            tf.append(new_val)
        idf = math.log(len(my_dict) / total)
        for x in tf:
            tf_idf.append(x * idf)
        d[keys[i]] = tf_idf

    df = pd.DataFrame(d)
    df.insert(0, 'Folder', folders)
    df.insert(0, 'File', files)
    return df


# part4
my_new_df = part3_tfidf(my_df)
part2_vis(my_new_df, 10)


def part_bonus(df):
    le = preprocessing.LabelEncoder()

    df['Folder'] = le.fit_transform(df['Folder'])
    df['File'] = le.fit_transform(df['File'])

    cols = [col for col in df.columns if col not in ["File", "Folder"]]
    data = df[cols]
    target = df['Folder']

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, random_state=40002)

    svc_model = LinearSVC(random_state=0, max_iter=10000)
    predict = svc_model.fit(data_train, target_train).predict(data_test)

    return "Accuracy:", accuracy_score(target_test, predict, normalize=True)


print(part_bonus(my_new_df))
