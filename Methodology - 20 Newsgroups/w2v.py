# Created by Tristan Harris, u17056145
# This program does the following
# 1. Text pre-processing
# 2. Word2vec Model training
# 3. Indexing reference and query sets using w2v word vectors
# 4. Compute soft cosine similarity matrix using ref & query sets
# 5. Plot the soft cosine similarities as histograms for inference

import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import re
import seaborn as sns
import time
from sklearn.datasets import fetch_20newsgroups
import scipy as sp
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
import multiprocessing
import pickle
import numpy as np
sns.set_style("darkgrid")

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

def apply_all(text):
    """
    Function that applies all the functions above into one
    Args:
        text: corpus

    Returns: preprocessed corpus

    """
    return initial_clean(text)

t1 = time.time()
df['tokenized'] = df['text'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(df), "articles:", (t2 - t1) / 60, "min")

# print(corpus[6])

# building the model
# instantiate model
cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=3,  # Ignore words that appear less than this
                     size=200,  # Dimensionality of word embeddings
                     workers=cores,  # Number of processors (parallelisation)
                     # sample=6e-5,
                     # alpha=0.03,
                     # min_alpha=0.0007,
                     window=5,  # Context window for words during training
                     # negative=20,  # negative sampling
                     iter=30,
                     sg=0)  # iterations

# generate a vocabulary
t = time.time()

w2v_model.build_vocab(df['tokenized'], progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))

# Train the model

t = time.time()

w2v_model.train(df['tokenized'], total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)

w2v_model.save("w2v-20newsgroups")

print(w2v_model.vector_size)

len(w2v_model.wv.vocab)

termsim_index = WordEmbeddingSimilarityIndex(w2v_model.wv)  # get termsim index
# dictionary = Dictionary(df['tokenized'])  # dictionary for model to use for indexing later
 # finding a similarity matrix
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
dictionary = load_obj('LDADICT')
bow_corpus = [dictionary.doc2bow(document) for document in df['tokenized']]  # generate a bow corpus
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)

# Testing: Case 1 Atheism vs Windows
# Basic Structure:
# Get a ref set indexed with w2v vocab
# Get query sets
# Check query against reference

def clean(cat, subset):
    """ some basic text preprocessing on a specified category of the 20 newsgroups dataset. Refer to sklean documentation for 20 newsgroups for potential categories.


    Args:
         cat - str: category from 20 newsgroups.
         type - str: train or test
    Return:
        gensim BOW in a list format
    """
    # type='train'
    # cat = 'alt.atheism'
    text = fetch_20newsgroups(categories=[cat], subset=subset, remove=('headers', 'footers', 'quote'))
    # Read the data into a pandas dataframe
    df = pd.DataFrame([text.data]).T
    df['text'] = df[0]
    df = df[df['text'].map(type) == str]
    df.dropna(axis=0, inplace=True, subset=['text'])
    df = df.sample(frac=1.0)
    df.reset_index(drop=True, inplace=True)
    corpus = df['text'].apply(apply_all)  # preprocessed tokenized corpus
    bow = [dictionary.doc2bow(doc) for doc in corpus]  # transform into a gen bow
    bow = [i for i in bow if len(i) > 0]  # remove empty lists
    return bow


train_ath = clean(cat='soc.religion.christian', subset='test')  # reference set
docsim_index = SoftCosineSimilarity(train_ath, similarity_matrix)  # SCM with ref set for later use

atheism_test = clean('soc.religion.christian', subset='test')  # query set 1: sim
windows_test = clean('comp.windows.x', subset='test')  # qyery set 2: diff

SCM_ath_ath = docsim_index[atheism_test]  # ref vs q1
SCM_ath_win = docsim_index[windows_test]  # ref vs q2

ath_ath = SCM_ath_ath.flatten()
ath_win = SCM_ath_win.flatten()
fig = plt.figure(1, figsize=(10, 5))
sns.distplot(ath_win, bins=50, color='red', label='Christian-WindowsX')  # atheism-windows
sns.distplot(ath_ath, bins=50, color='green', label='Christian-Christian')  # atheism-atheism

plt.ylabel('Frequency')
plt.xlabel('Soft Cosine Similarity')
fig.legend()
plt.title('Word2Vec: Christian Ref - Christian Query vs Christian Ref - WindowsX Query')
plt.tight_layout()
plt.savefig("w2v-chr-chrVSchr-windows.jpg")
fig.legend()
plt.show()

# Testing: Case 2 Atheism vs Christianity
#

train_ath = clean(cat='soc.religion.christian', subset='test')  # reference set
docsim_index = SoftCosineSimilarity(train_ath, similarity_matrix)  # SCM with ref set for later use

chr_test = clean('soc.religion.christian', subset='test')  # query set 1: sim
ath_test = clean('alt.atheism', subset='test')  # qyery set 2: sim

SCM_ath_chr = docsim_index[chr_test]  # ref vs q1
SCM_ath_ath = docsim_index[ath_test]  # ref vs q2


SCM_lat = docsim_index[train_ath]
sample_latex_SCM = pd.DataFrame(SCM_lat[0:5,0:5]).to_latex(header=False)
print(sample_latex_SCM)

ath_chr = SCM_ath_chr.flatten()
ath_ath_1 = SCM_ath_ath.flatten()

fig = plt.figure(1, figsize=(10, 5))
sns.distplot(ath_ath_1, bins=50, color='blue', label='Christian-Christian')  # atheism-atheism
sns.distplot(ath_chr, bins=50, color='orange', label='Atheism-Christian')  # atheism-windows
plt.ylabel('Frequency')
plt.xlabel('Soft Cosine Similarity')
fig.legend()
plt.title('Word2Vec: Christian - Atheism vs Christian - Christian')
plt.tight_layout()
plt.savefig("w2v-ath_ath_trainVSchr-atheism.jpg")
fig.legend()
plt.show()

# Testing: Case 3 Windows vs WindowsMisc
train = clean(cat='comp.windows.x', subset='test')  # reference set
docsim_index = SoftCosineSimilarity(train, similarity_matrix)  # SCM with ref set for later use

winx_test = clean('comp.windows.x', subset='test')  # query set 1: sim
windowsmisc_test = clean('comp.os.ms-windows.misc', subset='test')  # query set 2: sim

SCM_winxtest = docsim_index[winx_test]  # ref vs q1
SCM_winmisc = docsim_index[windowsmisc_test]  # ref vs q2

SCM_winxtest = SCM_winxtest.flatten()
SCM_winmisc = SCM_winmisc.flatten()

plt.figure(1, figsize=(12.5, 10))
sns.distplot(SCM_winxtest, bins=50, color='yellow', label='WindowsX-WindowsX')  # windowsx-windowsx
sns.distplot(SCM_winmisc, bins=50, color='violet', label='WindowsX-WindowsMisc')  # windowsMisc-windowsMisc

plt.ylabel('Frequency')
plt.xlabel('Soft Cosine Similarity')
plt.title('Word2Vec: WindowsX - WindowsX VS WindowsX - WindowsMisc')
plt.savefig("w2v-SCM_winxtest-SCM_winmisc.jpg")
plt.legend()
plt.show()

# Testing: Case 4 Christianity vs WindowsMisc and Windows
train_ath = clean(cat='comp.windows.x', subset='test')  # reference set
docsim_index = SoftCosineSimilarity(train_ath, similarity_matrix)  # SCM with ref set for later use

chr_test = clean('soc.religion.christian', subset='test')  # query set 1: diff
windowsmisc_test = clean('comp.os.ms-windows.misc', subset='test')  # qyery set 2: sim

SCM_winxtest_chr = docsim_index[chr_test]  # ref vs q1
SCM_winmisc = docsim_index[windowsmisc_test]  # ref vs q2

SCM_winxtest_chr = SCM_winxtest_chr.flatten()
SCM_winmisc = SCM_winmisc.flatten()

plt.figure(1, figsize=(10, 5))
sns.distplot(SCM_winxtest_chr, color='red', bins=50, label='WindowsX-Christian')  # Christianity-WindowsX
sns.distplot(SCM_winmisc, color='green', bins=50, label='WindowsX- WindowsMisc')  # Christianity - WindowsMisc

plt.ylabel('Frequency')
plt.xlabel('Soft Cosine Similarity')
plt.title('Word2Vec: WindowsX Ref - Christian - Query vs WindowsX Ref - WindowsMisc Query')
plt.savefig("w2v-chr-SCM_winxtestVSwin-SCM_winmisc.jpg")
plt.legend()
plt.show()

# subplots

fig, axs = plt.subplots(2, 1, figsize=(12.5,10))
sns.distplot(SCM_winxtest, bins=50, color='blue', ax=axs[0], label='WindowsX-WindowsX')  # windowsx-windowsx
sns.distplot(SCM_winmisc, bins=50, color='red', ax=axs[0], label='WindowsX-WindowsMisc')
sns.distplot(SCM_winxtest_chr, color='green', bins=50, ax=axs[1], label='WindowsX-Christian')  # Christianity-WindowsX
sns.distplot(SCM_winxtest, color='blue', bins=50, ax=axs[1], label='WindowsX-WindowsX')

[axs[i].set_ylabel('Frequency', fontsize='medium') for i in range(2)]
[axs[i].set_xlabel('Soft Cosine Similiarity', fontsize='medium') for i in range(2)]

axs[0].set_title('WindowsX Ref - WindowsX Query VS WindowsX Ref - WindowsMisc Query')
axs[1].set_title('WindowsX Ref- Christian - Query vs WindowsX Ref - WindowsMisc Query')
plt.tight_layout()
fig.subplots_adjust(top=0.88)
fig.legend(fontsize='medium')
plt.savefig('w2v_kde-windowsref_lda_dct.jpg', bbox_inches="tight")
plt.show()


fig, axs = plt.subplots(2, 1, figsize=(12.5,10))
sns.distplot(ath_ath, bins=50, color='yellow', ax=axs[0], label='Christian-Christian')  # windowsx-windowsx
sns.distplot(ath_win, bins=50, color='violet', ax=axs[0], label='Christian-WindowsX')
sns.distplot(ath_ath_1, color='red', bins=50, ax=axs[1], label='Christian-Christian')  # Christianity-WindowsX
sns.distplot(ath_chr, color='green', bins=50, ax=axs[1], label='Christian-Atheism')

[axs[i].set_ylabel('Frequency', fontsize='medium') for i in range(2)]
[axs[i].set_xlabel('Soft Cosine Similarity', fontsize='medium') for i in range(2)]

axs[0].set_title('Christian Ref - Christian Query VS Christian Ref - WindowsX Query')
axs[1].set_title('Christian Ref - Christian Query VS Christian Ref - Atheism Query')
plt.tight_layout()
# fig.subplots_adjust(top=0.88)
fig.legend(fontsize='medium')
plt.savefig('w2v_kde-chrisref_lda_dct.jpg', bbox_inches="tight")
plt.show()

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


save_obj(ath_ath, 'chr_chr_SCS')
save_obj(ath_win, 'chr_ath_SCS')
import random

sample_chr_w2v = clean(cat='soc.religion.christian', subset='test')# reference set

sample_chr_w2v = pd.DataFrame(sample_chr_w2v)
sample_chr_w2v.dropna(axis=0, inplace=True)
#%%
# msk = np.random.rand(len(sample_chr_w2v)) < 0.8
# sample_chr_w2v = sample_chr_w2v[~msk]
sample_chr_w2v.reset_index(drop=True,inplace=True)


def clean_testing(cat, subset, sample_size):
    """ some basic text preprocessing on a specified category of the 20 newsgroups dataset. Refer to sklean documentation for 20 newsgroups for potential categories.


    Args:
         cat - str: category from 20 newsgroups.
         type - str: train or test
    Return:
        gensim BOW in a list format
    """
    # type='train'
    # cat = 'alt.atheism'
    text = fetch_20newsgroups(categories=[cat], subset=subset, remove=('headers', 'footers', 'quote'))
    # Read the data into a pandas dataframe
    df = pd.DataFrame([text.data]).T
    df['text'] = df[0]
    df = df[df['text'].map(type) == str]
    df.dropna(axis=0, inplace=True, subset=['text'])
    df = df.sample(frac=1.0)
    df.reset_index(drop=True, inplace=True)
    corpus = df['text'].apply(apply_all) # preprocessed tokenized corpus
    corpus = np.random.choice(corpus, size=sample_size, replace=False)
    bow = [dictionary.doc2bow(doc) for doc in corpus]  # transform into a gen bow
    bow = [i for i in bow if len(i) > 0]  # remove empty lists
    return bow


chr_sample = clean_testing("soc.religion.christian", "test", 100)
win_sample = clean_testing('comp.windows.x', 'test', 100)

docsim_index = SoftCosineSimilarity(chr_sample, similarity_matrix)  # SCM with ref set for later use


SCM_win_chr_sample = docsim_index[win_sample]
scm_chr_chr_sample = docsim_index[chr_sample]  # ref vs q1


SCM_chr_chr_sample = scm_chr_chr_sample.flatten()
SCM_win_chr_sample = SCM_win_chr_sample.flatten()

save_obj(SCM_chr_chr_sample, 'chr_chr_SCS_sample')
save_obj(SCM_win_chr_sample, 'chr_ath_SCS_sample')
