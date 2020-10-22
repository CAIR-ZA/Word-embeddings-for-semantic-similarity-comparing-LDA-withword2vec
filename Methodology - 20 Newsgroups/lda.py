# Created by Tristan Harris, u17056145
# This program does the following
# 1. Text pre-processing, create dictionary and convert corpus to BOW
# 2. LDA Model training
# 3. Indexing reference and query sets using the trained model and dictionary
# 4. Compute Jensen-Shannon distances matrix using ref & query sets
# 5. Plot the Jensen-Shannon distances as histograms for inference

import pandas as pd
import numpy as np
from nltk.corpus import stopwords

from gensim.models import LdaModel
from gensim import corpora
import re
from nltk.stem.porter import PorterStemmer
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
import scipy as sp
import nltk
import pickle
from scipy.spatial.distance import jensenshannon

nltk.download('stopwords')  # run only once

sns.set_style("white")

# Retrieve data from the sklean database and partially preprocess
# Use entire training dataset of sklearn
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quote'))

# real data lies in filenames and target attributes
# target attribute is the integer index of the category

print("Corpora of interest:", newsgroups_train.target_names)
print(newsgroups_train.filenames.shape)
arr = np.flip(np.array(newsgroups_train.data))
print(arr)
# Read the data into a pandas dataframe
df = pd.DataFrame([newsgroups_train.data]).T
df['text'] = df[0]

print(df)
df = df[df['text'].map(type) == str]
print(df)
df.dropna(axis=0, inplace=True, subset=['text'])
print(df)
df = df.sample(frac=1.0)
df.reset_index(drop=True, inplace=True)
df.head()


def initial_clean(text):
    """
    Function to clean text of websites, email addressess and any punctuation
    We also lower case the text
    Args:
        text: raw corpus

    Returns: tokenized corpus

    """
    text = re.sub(r"((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    return text


stop_words = stopwords.words('english')


def remove_stop_words(text):
    """
    Function that removes all stopwords from tokenized corpus
    Args:
        text: corpus

    Returns: corpus w/o stopwords

    """
    return [word for word in text if word not in stop_words]


stemmer = PorterStemmer()


def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    Args:
        text: corpus

    Returns: corpus processed so that plural and singular words are treated the same

    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1]  # make sure we have no single letter words
    except IndexError:  # the word "oed" broke this, so needed to try except
        pass
    return text


def apply_all(text):
    """
    Function that applies all the functions above into one
    Args:
        text: corpus

    Returns: preprocessed corpus

    """
    return stem_words(remove_stop_words(initial_clean(text)))


# clean text and title and create new column "tokenized"
t1 = time.time()
df['tokenized'] = df['text'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(df), "articles:", (t2 - t1) / 60, "min")

df['doc_len'] = df['tokenized'].apply(lambda x: len(x))
doc_lengths = list(df['doc_len'])
df.drop(labels='doc_len', axis=1, inplace=True)

print("length of list:", len(doc_lengths),
      "\naverage document length", np.average(doc_lengths),
      "\nminimum document length", min(doc_lengths),
      "\nmaximum document length", max(doc_lengths))

# LDA doesnt work very well with short documents
df = df[df['tokenized'].map(len) >= 40]  # If this value changes the JS distances look different
df = df[df['tokenized'].map(type) == list]
df.reset_index(drop=True, inplace=True)
print("After cleaning and excluding short articles the dataframe now has:", len(df), "articles")

train_df = df
train_df.reset_index(drop=True, inplace=True)

print(train_df)

stop_words = stopwords.words('english')

dct = corpora.Dictionary(train_df['tokenized'])  # dictionary for indexing


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


save_obj(dct, 'LDADICT')
corpus = [dct.doc2bow(line) for line in train_df['tokenized']]  # bowCorpus
save_obj(corpus, 'LDABOWcorpus')

num_topics = 20  # topics declared
chunk_size = 300
t1 = time.time()
# low alpha => each doc only rep by small num topics & vice versa
# low eta means each topic only rep by small num words and vice versa
lda = LdaModel(corpus=corpus,
               num_topics=num_topics,
               id2word=dct,
               alpha='auto',
               random_state=100,
               # eta=None,
               update_every=1,
               chunksize=chunk_size,
               minimum_probability=0.0,
               # iterations=100,
               # gamma_threshold=0.001,
               passes=10,
               per_word_topics=True)

lda.get_document_topics(bow=corpus, per_word_topics=True)
tpl = lda.print_topics(num_topics=20, num_words=5)
topic, contrib = zip(*tpl)

t2 = time.time()
print("Time to train LDA model on", len(df), "articles:", (t2 - t1) / 60, "min")
top_k_topics = lda.top_topics(corpus, topn=5, dictionary=dct, texts=train_df['tokenized'])
indx = [i + 1 for i in range(20)]
contrib = np.transpose(contrib)

DTdist = pd.DataFrame(contrib, columns=["Top 5 words that contribute to each topic with associated probability"],
                      index=indx)

distLatex = DTdist.to_latex(index=True, index_names="Topics")
# document distribution
doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=corpus, per_word_topics=False)])
obj = lda.get_topics()
a = lda.inference(corpus)
print(doc_distribution[:853])
# training corpus document by topic matrix
doc_topic_dist_corpus = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])


# TESTING

def clean(cat):
    """
    Carry out some basic text preprocessing on a specified category of the 20 newsgroups dataset
    params: cat - str: category from 20 newsgroups. Refer to sklean documentation for 20 newsgroups.
    """
    text = fetch_20newsgroups(categories=[cat], subset='test', remove=('headers', 'footers', 'quote'))
    # Read the data into a pandas dataframe
    df = pd.DataFrame([text.data]).T
    df['text'] = df[0]
    df = df[df['text'].map(type) == str]
    df.dropna(axis=0, inplace=True, subset=['text'])
    df = df.sample(frac=1.0)
    df.reset_index(drop=True, inplace=True)
    df['tokenized'] = df['text'].apply(apply_all)
    df = df[df['tokenized'].map(len) >= 40]  # If this value changes the JS distances look different
    df = df[df['tokenized'].map(type) == list]
    df.reset_index(drop=True, inplace=True)
    return df


# Reference and query sets
Windows = clean('comp.windows.x')  # ref
WindowsMisc = clean('comp.os.ms-windows.misc')  # query
christian_sci = clean('soc.religion.christian')  # ref
atheism_sci = clean('alt.atheism')  # query



def doc_topic_dist(data):
    """
    Creates a document by topic distribution for a specified category using the trained LDA model
    """
    bowCorpus = [dct.doc2bow(line) for line in data['tokenized']]
    doc_topic_distro = np.array([[tup[1] for tup in lst] for lst in lda[bowCorpus]])
    return doc_topic_distro


doc_topic_dist_Windows = doc_topic_dist(Windows)
doc_topic_dist_christian = doc_topic_dist(christian_sci)
doc_topic_dist_WinMisc = doc_topic_dist(WindowsMisc)
doc_topic_dist_atheism = doc_topic_dist(atheism_sci)



def jsd_mat(p, q):
    """
    Generates a matrix of JSDs using the document topic distribution matrices. Not computationally efficient for large
    vectors of p and q (>1000)


    Args:
        p: document-topic distribution matrix
        q: document-topic distribution matrix
    Return:
        matrix: a len(p) by len(q) matrix of JSDs
    """

    n_p = len(p)
    n_q = len(q)
    matrix = np.zeros((n_p, n_q))
    for j in range(n_q):
        for i in range(n_p):
            matrix[i, j] = 1 - jensenshannon(p[i][:], q[j][:])  # ADJUST JSD HERE
    return matrix



js_win_win = jsd_mat(doc_topic_dist_Windows, doc_topic_dist_Windows)  # ref1 v ref1
js_christian_christian = jsd_mat(doc_topic_dist_christian, doc_topic_dist_christian)  # ref2 v ref2
js_chr_ath = jsd_mat(doc_topic_dist_christian, doc_topic_dist_atheism)  # ref1 v q1
js_comp = jsd_mat(doc_topic_dist_Windows, doc_topic_dist_WinMisc)  # ref2 v q2
JS_Windows_Christianity = jsd_mat(doc_topic_dist_Windows, doc_topic_dist_christian)  # ref1 v ref2
JS_Christianity_Windows = jsd_mat(doc_topic_dist_christian, doc_topic_dist_Windows)  # ref2 v ref1


win2 = doc_topic_dist_Windows[37:39]
christian1 = doc_topic_dist_christian[255:256]
doc_dis = np.concatenate([win2, christian1])
win_test = doc_topic_dist_Windows[99:100]
JS_Windows_ChristianityToy = jsd_mat(doc_dis, win_test)

sample_JSD = js_christian_christian[0:5, 0:5]  # sample for display purposes
sample_JSD = pd.DataFrame(sample_JSD).to_latex()


# TEST PLOTS

js_win_win = js_win_win[js_win_win != 1]  # remove any 0 values
js_christian_christian = js_christian_christian[js_christian_christian != 1]  # remove any 0 values

# flatten
js_win_win = np.transpose(js_win_win.flatten())
js_christian_christian = np.transpose(js_christian_christian.flatten())
JS_Christianity_Windows = np.transpose(JS_Christianity_Windows.flatten())
JS_Windows_Christianity = np.transpose(JS_Windows_Christianity.flatten())
js_comp = np.transpose(js_comp.flatten())
js_chr_ath = np.transpose(js_chr_ath.flatten())
d1 = [js_win_win, JS_Christianity_Windows]
d2 = [JS_Windows_Christianity, js_christian_christian]
data = np.mat([d1, d2])

means = [np.mean(data[i, j]) for i in range(2) for j in range(2)]
median = [np.median(data[i, j]) for i in range(2) for j in range(2)]
sum_stats_LaTeX = pd.DataFrame([means, median],
                               columns=['Windows vs Windows', 'Christianity vs Windows', 'Windows vs Christianity',
                                        'Christianity vs Christianity'],
                               index=['Mean', 'Median']).to_latex(index=True)
print(means)


def js_plots(data, labels, color=None, **kwargs):
    """
    Generates kde plots of jensen-shannon distances for categories specifed

    Args:
        data: list of two JSDs
        labels: list of category names
        color: list of 2 colors for plots
    Returns:
        KDE plot of JSDs on same axis
    """

    if color is None:
        color = ['red', 'blue']
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(1, 1, figsize=(12.5, 10))

    [sns.distplot(data[i], ax=axs, kde=True, color=wd, **kwargs) for i, wd in enumerate(color)]
    axs.set_ylabel('Frequency', fontsize='medium')
    axs.set_xlabel('Jensen-Shannon Distances', fontsize='medium')

    axs.set_title('Jensen-Shannon distances of ' + labels[0] + ' and ' + labels[1],
                  fontsize='large')
    plt.tight_layout()
    # fig.subplots_adjust(top=0.88)
    fig.legend(labels=labels, fontsize='medium')
    plt.savefig('LDA_kde' + labels[0] + labels[1] + '.jpg', bbox_inches="tight")

    plt.show()


col = ['blue', 'green']
l1 = ['Windows-Windows', 'Christian-Windows']
js_plots(d1, l1, col, bins=50)

d0 = [js_christian_christian, js_chr_ath]
l2 = ['Christian-Christian', 'Christian-Atheism']
js_plots(d0, l2, bins=50)

d3 = [js_win_win, js_comp]
l3 = ['Windows-Windows', 'MiscWindows-Windows']
c3 = ['red', 'green']
js_plots(d3, l3, c3, bins=50)

d4 = [js_christian_christian, JS_Christianity_Windows]
l4 = ['Christian-Christian', 'Christian-Windows']
c4 = ['orange', 'violet']
js_plots(d4, l4, c4, bins=50)

data = [d1, d2, d3]
meansd1 = np.array([np.mean(d1[i]) for i in range(len(d1))])
meansd2 = [np.mean(d2[i]) for i in range(len(d2))]
meansd3 = [np.mean(d3[i]) for i in range(len(d3))]

meansd1_LaTeX = pd.DataFrame(list(meansd1), columns=['Mean'], index=l1).to_latex(index=True)
meansd2_LaTeX = pd.DataFrame(meansd2, index=l2, columns=['Mean']).to_latex(index=True)
meansd3_LaTeX = pd.DataFrame(meansd3, index=l3, columns=['Mean']).to_latex(index=True)


def js_plots_stacked(d1, d2, l1, l2, p1=None, p2=None, **kwargs):
    """
    Generates subplot kde plots of jensen-shannon distances for categories specifed stacked.
    Reference sets intended to be the same colors.
    Args:
        d1: list of two JSDs
        d2: list of two JSDs
        l1: list of category names for d1
        l2: list of category names for d2
        p1: colors for d1. Default = ['orange', 'blue']
        p2: colors for d2. Default = ['orange', 'green']
    Returns:
        sns.distplot() plot of JSDs on same axis on subplots
    """

    if p1 is None:
        p1 = ['orange', 'blue']
    if p2 is None:
        p2 = ['orange', 'green']
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 10))

    [sns.distplot(d1[i], ax=axs[0], kde=True, label=l1[i], color=wd, **kwargs) for i, wd in enumerate(p1)]
    [sns.distplot(d2[j], ax=axs[1], kde=True, label=l2[j], color=wd, **kwargs) for j, wd in enumerate(p2)]

    [axs[i].set_ylabel('Frequency', fontsize='medium') for i in range(2)]
    [axs[i].set_xlabel('Jensen-Shannon Distances (Adjusted)', fontsize='medium') for i in range(2)]

    axs[0].set_title('Adjusted Jensen-Shannon distances of ' + l1[0] + ' and ' + l1[1],
                     fontsize='large')
    axs[1].set_title('Adjusted Jensen-Shannon distances of ' + l2[0] + ' and ' + l2[1],
                     fontsize='large')
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.legend(fontsize='medium')
    plt.savefig('LDA_kde' + l1[0] + l1[1] + '-' + l2[0] + l2[1] + '.jpg', bbox_inches="tight")

    plt.show()


d0 = [js_christian_christian[js_christian_christian != 1], js_chr_ath[js_chr_ath != 1]]
l2 = ['Christian-Christian', 'Christian-Atheism']
d4 = [js_christian_christian[js_christian_christian != 1], JS_Christianity_Windows[JS_Christianity_Windows != 1]]
l4 = ['Christian-Christian', 'Christian-Windows']
labels = [l2, l4]

js_plots_stacked(d0, d4, l2, l4, bins=50)

d3 = [js_win_win[js_win_win != 1], js_comp[js_comp != 1]]
l3 = ['Windows-Windows', 'Windows-MiscWindows']
c3 = ['red', 'green']
d1 = [js_win_win[js_win_win != 1], JS_Christianity_Windows[JS_Christianity_Windows != 1]]
col = ['red', 'blue']
l1 = ['Windows-Windows', 'Windows-Christian']
js_plots_stacked(d3, d1, l3, l1, c3, col, bins=50)

save_obj(JS_Windows_Christianity, 'jsd_win_chr')
save_obj(js_christian_christian, 'jsd_chr_chr')


# Testing
msk = np.random.rand(len(christian_sci)) < 0.8

#%%

sample_df = np.random.choice(christian_sci['tokenized'], 100)
sample_df = pd.DataFrame(sample_df, columns=['tokenized'])

sample_DTD_chr=doc_topic_dist(sample_df)
sample_chr=jsd_mat(sample_DTD_chr, sample_DTD_chr)
sample_chr = sample_chr.flatten()
save_obj(sample_chr, 'sample_chr_chr')


msk = np.random.rand(len(Windows)) < 0.8

#%%
sample_df_win = np.random.choice(Windows['tokenized'], 100)
sample_df_win = pd.DataFrame(sample_df_win, columns=['tokenized'])


sample_DTD_win=doc_topic_dist(sample_df_win)
sample_win=jsd_mat(sample_DTD_win, sample_DTD_chr)
sample_win = sample_win.flatten()
save_obj(sample_win, 'sample_win_chr')

