# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:01:32 2020

@author: Tristan Harris
"""
#%%


import nltk
import matplotlib.pyplot as plt

import seaborn as sns
import time

import scipy as sp
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
import multiprocessing
import pickle

sns.set_style("darkgrid")



from gensim.models import LdaModel
from gensim import corpora

from nltk.stem.porter import PorterStemmer




import sys 
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess


# NLTK Stop words
from nltk.corpus import stopwords

warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def save_obj(obj, name):
    """
    Save a data-type of interest as a .pkl file

    Args:
        obj (any): variable name of interest
        name (str): string-name for .pkl file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Load .pkl file from current working directory

    Args:
        name (str): name for .pkl file of interest

    Returns:
        [any]: unpacked .pkl file either in the form of a pd.DataFrame or list
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# nltk.download('stopwords')  # run only once

sns.set_style("white")

# Retrieve data from the sklean database and partially preprocess
# Use entire training dataset of sklearn



# real data lies in filenames and target attributes
# target attribute is the integer index of the category

df = load_obj('df_application')

#%%
# be aware cell takes approximately 45 mins to run
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

w2v_model.build_vocab(df['Text'], progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))

# Train the model

t = time.time()

w2v_model.train(df['Text'], total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)

w2v_model.save("w2v-application")

print(w2v_model.vector_size) 

len(w2v_model.wv.vocab)

termsim_index = WordEmbeddingSimilarityIndex(w2v_model.wv)  # get termsim index
# dictionary = Dictionary(df['tokenized'])  # dictionary f
dictionary = load_obj('LDADICT-application')
bow_corpus = [dictionary.doc2bow(document) for document in df['Text']]  # generate a bow corpus
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)

#%%
df_test_dominant_topic=load_obj('application-testset')
test0=df_test_dominant_topic['Dominant_Topic']==0
test1=df_test_dominant_topic['Dominant_Topic']==1
test2=df_test_dominant_topic['Dominant_Topic']==2
test3=df_test_dominant_topic['Dominant_Topic']==3
test4=df_test_dominant_topic['Dominant_Topic']==4
test5=df_test_dominant_topic['Dominant_Topic']==5
df_test0=df_test_dominant_topic[test0]
df_test0.reset_index(drop=True, inplace=True)
df_test1=df_test_dominant_topic[test1]
df_test1.reset_index(drop=True, inplace=True)
df_test2=df_test_dominant_topic[test2]
df_test2.reset_index(drop=True, inplace=True)
df_test3=df_test_dominant_topic[test3]
df_test3.reset_index(drop=True, inplace=True)
df_test4=df_test_dominant_topic[test4]
df_test4.reset_index(drop=True, inplace=True)
df_test6=df_test_dominant_topic[test5]
df_test6.reset_index(drop=True, inplace=True)

#%%


def get_bow(corpus):
    """ some basic text preprocessing on a specified category of the 20 newsgroups dataset. Refer to sklean documentation for 20 newsgroups for potential categories.


    Args:
        corpus: (List) Tokenized data
    Return:
        gensim BOW in a list format
    """
    # type='train'
    # preprocessed tokenized corpus
    bow = [dictionary.doc2bow(doc) for doc in corpus]  # transform into a gen bow
    bow = [i for i in bow if len(i) > 0]  # remove empty lists
    return bow
#%%
test1BOW=get_bow(df_test1['tokenized'])
train1=df['Dominant_Topic']==1
dftrain1=df[train1]
dftrain1.reset_index(drop=True, inplace=True)
train1BOW=get_bow(dftrain1['Text'])
#%%

testvstraining_1=SoftCosineSimilarity(train1BOW,similarity_matrix)
scs_topic1=testvstraining_1[test1BOW]
#%%

test4BOW=get_bow(df_test4['tokenized'])
scs_topic1vstopic4=testvstraining_1[test4BOW]


#%%
fig = plt.figure(1, figsize=(10, 5))
sns.distplot(scs_topic1, bins=50, color='red', label='TrainingvsTest (Topic1)')  # atheism-windows
sns.distplot(scs_topic1vstopic4, bins=50, color='green', label='Topic1vsTopic4')  # atheism-atheism

plt.ylabel('Frequency')
plt.xlabel('Soft Cosine Similarity')
fig.legend()
plt.title('Similarity and Disimilarity Semantic Testing')
plt.tight_layout()
plt.savefig("w2v-application1.jpg")
fig.legend()
plt.show()


#%%

# comparing topic 4: training vs testing
test4BOW=get_bow(df_test4['tokenized'])
train4=df['Dominant_Topic']==4
dftrain4=df[train4]
dftrain4.reset_index(drop=True, inplace=True)
train4BOW=get_bow(dftrain4['Text'])
#%%

testvstraining_4=SoftCosineSimilarity(train4BOW,similarity_matrix)
scs_topic4=testvstraining_4[test4BOW]
#%%
# comparing topic 4 (training) vs topic 6 (test)
test6BOW=get_bow(df_test6['tokenized'])
scs_topic6vstopic4=testvstraining_4[test6BOW]

#%%
fig = plt.figure(1, figsize=(10, 5))
sns.distplot(scs_topic4, bins=50, color='blue', label='TrainingvsTest (Topic4)')  # atheism-windows
sns.distplot(scs_topic6vstopic4, bins=50, color='orange', label='Topic6vsTopic4')  # atheism-atheism
plt.ylabel('Frequency')
plt.xlabel('Soft Cosine Similarity')
fig.legend()
plt.title('Similarity and Disimilarity Semantic Testing')
plt.tight_layout()
plt.savefig("w2v-application2.jpg")
fig.legend()
plt.show()


#%%



def scs_plots_stacked(d1, d2, l1, l2, p1=None, p2=None, **kwargs):
    """
    Generates subplot kde plots of soft-cosine similarity for categories specifed stacked.
    Reference sets intended to be the same colors.
    Args:
        d1: list of two JSDs
        d2: list of two JSDs
        l1: list of category names for d1
        l2: list of category names for d2
        p1: colors for d1. Default = ['orange', 'blue']
        p2: colors for d2. Default = ['red', 'green']
    Returns:
        sns.distplot() plot of JSDs on same axis on subplots
    """

    if p1 is None:
        p1 = ['orange', 'blue']
    if p2 is None:
        p2 = ['red', 'green']
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 10))

    [sns.distplot(d1[i], ax=axs[0], kde=True, label=l1[i], color=wd, **kwargs) for i, wd in enumerate(p1)]
    [sns.distplot(d2[j], ax=axs[1], kde=True, label=l2[j], color=wd, **kwargs) for j, wd in enumerate(p2)]

    [axs[i].set_ylabel('Frequency', fontsize='medium') for i in range(2)]
    [axs[i].set_xlabel('Jensen-Shannon Distances (Adjusted)', fontsize='medium') for i in range(2)]

    axs[0].set_title('Soft-Cosine Similarity Comparisons of ' + l1[0] + ' and ' + l1[1],
                     fontsize='large')
    axs[1].set_title('Soft-Cosine Similarity Comparisons of ' + l2[0] + ' and ' + l2[1],
                     fontsize='large')
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.legend(fontsize='medium')
    plt.savefig('SCS_kde' + l1[0] + l1[1] + '-' + l2[0] + l2[1] + '-application.jpg', bbox_inches="tight")

    plt.show()

scs_plots_stacked([scs_topic1, scs_topic1vstopic4],[scs_topic4, scs_topic6vstopic4], ['Virology (Testing vs Training)', 'Virology (Testing) vs Genetics (Training)'], ['Genetics (Testing vs Training)', 'Pathology (Testing) vs Genetics (Training)'])

#%%
def get_ref_query(corpus):
    """
    Carry out some basic text preprocessing on a specified category of the 20 newsgroups dataset
    params: cat - str: category from 20 newsgroups. Refer to sklean documentation for 20 newsgroups.
    """
 # preprocessed tokenized corpus
    msk = np.random.rand(len(corpus)) < 0.80
    ref_df = corpus[msk]
    ref_df.reset_index(drop=True, inplace=True)
    query_df = corpus[~msk]
    query_df.reset_index(drop=True, inplace=True)
    bow_r = [dictionary.doc2bow(doc) for doc in ref_df]  # transform into a gen bow
    bow_r = [i for i in bow_r if len(i) > 0]
    bow_q = [dictionary.doc2bow(doc) for doc in query_df]  # transform into a gen bow
    bow_q = [i for i in bow_q if len(i) > 0]  # remove empty lists
    return bow_r, bow_q


#%%
# virology 
top1_ref, top1_query = get_ref_query(df_test1['tokenized'])
# genetics
top4_ref, top4_query = get_ref_query(df_test4['tokenized'])


#%%
docsim_index = SoftCosineSimilarity(top4_ref, similarity_matrix)  # SCM with ref set for later use

scm_self = docsim_index[top4_ref]
ref = scm_self[~np.eye(scm_self.shape[0], dtype=bool)].reshape(scm_self.shape[0], -1)
means = np.mean(ref, axis=0)
sns.distplot(means, bins=30, color='purple')
plt.xlabel('SCS Means')
plt.ylabel('Frequency')
plt.show()

#%%

SCM_sim = docsim_index[top4_query]
SCM_sim = np.mean(SCM_sim, axis=1)
plt.hist(SCM_sim)
plt.xlabel('SCS Means')
plt.ylabel('Frequency')
plt.title('Distribution of Means of Query-Ref (Similar) using Topic 5: Genetics')
plt.show()

#%%
prob_sim = []
for j in range(len(SCM_sim)):
    count2 = [i for i in means if i <= SCM_sim[j]]
    prob_sim.append(len(count2) / len(means))
    
plt.hist(prob_sim, bins=30)
plt.title('Probabilities of Means of New Docs (Similar) inside generated distribution')
plt.ylabel('Frequency')
plt.xlabel('Probability')
plt.show()

save_obj(prob_sim, 'probsim_w2v-application')

#%%

scm_diff = docsim_index[top4_query]
scm_diff = np.mean(scm_diff, axis=1)
plt.hist(scm_diff)
plt.xlabel('SCS Means')
plt.ylabel('Frequency')
plt.title('Distribution of Means of Query1-Ref (Disim) (70 unseen docs)')
plt.show()

#%%
prob_disim = []
for j in range(len(scm_diff)):
    count1 = [i for i in means if i <= scm_diff[j]]
    prob_disim.append(len(count1) / len(means))
    
plt.hist(prob_disim, bins=30)
plt.title('Probabilities of Means of New Docs (Similar) inside generated distribution')
plt.ylabel('Frequency')
plt.xlabel('Probability')
plt.show()

save_obj(prob_disim, 'probdisim_w2v-application')


