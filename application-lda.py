#%%


import pandas as pd
import numpy as np


from gensim.models import LdaModel
from gensim import corpora
import re
from nltk.stem.porter import PorterStemmer
import time
import seaborn as sns
import nltk
import pickle
from wordcloud import WordCloud

import sys 
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
import matplotlib.pyplot as plt

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
data = load_obj('training_application')


# real data lies in filenames and target attributes
# target attribute is the integer index of the category
#%%

# Read the data into a pandas dataframe
df = pd.DataFrame(data)
df['text'] = df[0]
print(df)
df = df[df['text'].map(type) == str]
print(df)
df.dropna(axis=0, inplace=True, subset=['text'])
print(df)
df = df.sample(frac=1.0)
df.reset_index(drop=True, inplace=True)
df.head()


# %%

# Cleaning function definitions
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
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come', 'et', 'al', 'without', 'use', 'figur', 'howev'])



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
#%%

# may take some time: ETA: 23 minutes
# clean text and title and create new column "tokenized"
t1 = time.time()
df['tokenized'] = df['text'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(df), "articles:", (t2 - t1) / 60, "min")

#%%
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


#%%
df['doc_len'] = df['tokenized'].apply(lambda x: len(x))
doc_lengths = list(df['doc_len'])
df.drop(labels='doc_len', axis=1, inplace=True)

print("length of list:", len(doc_lengths),
      "\naverage document length", np.average(doc_lengths),
      "\nminimum document length", min(doc_lengths),
      "\nmaximum document length", max(doc_lengths))
train_df = df
train_df.reset_index(drop=True, inplace=True)


dct = corpora.Dictionary(train_df['tokenized'])  # dictionary for indexing
save_obj(df, "df-cleaned-application")
save_obj(dct, 'LDADICT-application')
corpus = [dct.doc2bow(line) for line in train_df['tokenized']]  # bowCorpus
save_obj(corpus, 'LDABOWcorpus-application')
#%%

# load cleaned dataset (ETA: <1min)
df=load_obj('df-cleaned-application')
df.reset_index(drop=True, inplace=True)
dct = load_obj('LDADICT-application')
corpus = load_obj('LDABOWcorpus-application') 
#%%
# lda model training  (ETA 10 mins)

num_topics = 6  # topics declared (based on MATLAB tut)
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
tpl = lda.print_topics(num_topics=6, num_words=5)
topic, contrib = zip(*tpl)

t2 = time.time()
print("Time to train LDA model on", len(df), "articles:", (t2 - t1) / 60, "min")

top_k_topics = lda.top_topics(corpus, topn=5, dictionary=dct, texts=train_df['tokenized'])
indx = [i + 1 for i in range(6)]
contrib = np.transpose(contrib)
#%%
tpl = lda.print_topics(num_topics=6, num_words=5)
topic, contrib = zip(*tpl)
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
save_obj(lda, 'LDA_MODEL_APPLICATION')
#%%
lda = load_obj('LDA_MODEL_APPLICATION')
fig, axes = plt.subplots(2, 3, figsize=(20,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    plt.imshow(WordCloud(background_color="white").fit_words(dict(lda.show_topic(i, 200))))
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()




#%%
# finding dominant topics in the corpus for each document
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

#%%
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda, corpus=corpus, texts=df['tokenized'])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)
#%%
# save_obj(df_dominant_topic,'df_application')
lda = load_obj('LDA_MODEL_APPLICATION')
#%%

top0=df_dominant_topic['Dominant_Topic']==0
top1=df_dominant_topic['Dominant_Topic']==1
top2=df_dominant_topic['Dominant_Topic']==2
top3=df_dominant_topic['Dominant_Topic']==3
top4=df_dominant_topic['Dominant_Topic']==4
top6=df_dominant_topic['Dominant_Topic']==5
df_top0=df_dominant_topic[top0]
df_top0.reset_index(drop=True, inplace=True)
df_top1=df_dominant_topic[top1]
df_top1.reset_index(drop=True, inplace=True)
df_top2=df_dominant_topic[top2]
df_top2.reset_index(drop=True, inplace=True)
df_top3=df_dominant_topic[top3]
df_top3.reset_index(drop=True, inplace=True)
df_top4=df_dominant_topic[top4]
df_top4.reset_index(drop=True, inplace=True)
df_top6=df_dominant_topic[top6]
df_top6.reset_index(drop=True, inplace=True)

#%%
# jensen shannon distance testing

def doc_topic_dist(data):
    """
    Creates a document by topic distribution for a specified category using the trained LDA model
    """
    bowCorpus = [dct.doc2bow(line) for line in data['Text']]
    doc_topic_distro = np.array([[tup[1] for tup in lst] for lst in lda[bowCorpus]])
    return doc_topic_distro


dtd_topic0=doc_topic_dist(df_top0)
dtd_topic1=doc_topic_dist(df_top1)
dtd_topic2=doc_topic_dist(df_top2)
dtd_topic3=doc_topic_dist(df_top3)
dtd_topic4=doc_topic_dist(df_top4)
dtd_topic6=doc_topic_dist(df_top6)
#%%
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
    from scipy.spatial.distance import jensenshannon

    n_p = len(p)
    n_q = len(q)
    matrix = np.zeros((n_p, n_q))
    for j in range(n_q):
        for i in range(n_p):
            matrix[i, j] = 1 - jensenshannon(p[i][:], q[j][:])  # ADJUST JSD HERE
    return matrix
# 10 mins or so
top0_top1_jsd=jsd_mat(dtd_topic1, dtd_topic0)

#%%
t1=time.time()
T1T4_JSD=jsd_mat(dtd_topic1, dtd_topic4)
t2 = time.time()
print("Time to calculate JSD Matrix:", (t2 - t1) / 60, "min")
#%%
t1=time.time()
T1T4_JSD=jsd_mat(dtd_topic1, dtd_topic3)
t2 = time.time()
print("Time to calculate JSD Matrix:", (t2 - t1) / 60, "min")
#%%
t1=time.time()
top4_top4_jsd=jsd_mat(dtd_topic1, dtd_topic1)
t2 = time.time()
print("Time to calculate JSD Matrix:", (t2 - t1) / 60, "min")
#%%
# plotting 
# T1T4=np.transpose(T1T4_JSD.flatten())
# T0T1=np.transpose(top0_top0_jsd.flatten())
# T4T4=np.transpose(top4_top4_jsd.flatten())
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
        color = ['orange', 'blue']
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(1, 1, figsize=(12.5, 10))

    [sns.distplot(data[i], ax=axs, kde=True, color=wd, **kwargs) for i, wd in enumerate(color)]
    axs.set_ylabel('Frequency', fontsize='medium')
    axs.set_xlabel('Jensen-Shannon Distances', fontsize='medium')

    axs.set_title('Jensen-Shannon distances of ' + labels[0] + ' and ' + labels[1],
                  fontsize='large')
    plt.tight_layout()
    # fig.subplots_adjust(top=0.88)
    fig.legend(labels=labels, fontsize='medium', loc="right")
    plt.savefig('LDA_kde' + labels[0] + labels[1] + '.jpg', bbox_inches="tight")

    plt.show()
#js_plots([T1T4,T4T4], ["Virology vs Pathology","Pathology (Self)"])

#%%
# test dataset
data = load_obj('test_data_application')

   
# Read the data into a pandas dataframe
df = pd.DataFrame(data)
df['text'] = df[0]

df = df[df['text'].map(type) == str]

df.dropna(axis=0, inplace=True, subset=['text'])

df = df.sample(frac=1.0)
df.reset_index(drop=True, inplace=True)
test=df
test.reset_index(drop=True, inplace=True)
#%% 
def doc_topic_dist_corpus(data):
    """
    Creates a document by topic distribution for a specified category using the trained LDA model
    """
    bowCorpus = [dct.doc2bow(line) for line in data['tokenized']]
    doc_topic_distro = np.array([[tup[1] for tup in lst] for lst in lda[bowCorpus]])
    return doc_topic_distro, bowCorpus

#%%
# DTD of test set

t1 = time.time()
test['tokenized'] = test['text'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(test), "articles:", (t2 - t1))
# testDTD=doc_topic_dist(test)

test['doc_len'] = test['tokenized'].apply(lambda x: len(x))
doc_lengths = list(test['doc_len'])
test.drop(labels='doc_len', axis=1, inplace=True)

print("length of list:", len(doc_lengths),
      "\naverage document length", np.average(doc_lengths),
      "\nminimum document length", min(doc_lengths),
      "\nmaximum document length", max(doc_lengths))

# LDA doesnt work very well with short documents
test = test[test['tokenized'].map(len) >= 40]  # If this value changes the JS distances look different
test = test[test['tokenized'].map(type) == list]
test.reset_index(drop=True, inplace=True)
print("After cleaning and excluding short articles the dataframe now has:", len(test), "articles")

#%%
bowCorpus = [dct.doc2bow(line) for line in test['tokenized']]
#%%
df_test = format_topics_sentences(ldamodel=lda, corpus=bowCorpus, texts=test['tokenized'])
#%%
df_test_dominant_topic = df_test.reset_index()
df_test_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_test_dominant_topic.head(10)
save_obj(df_test, 'application-testset')

#%%

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
def doc_topic_dist(data):
    """
    Creates a document by topic distribution for a specified category using the trained LDA model
    """
    bowCorpus = [dct.doc2bow(line) for line in data['Text']]
    doc_topic_distro = np.array([[tup[1] for tup in lst] for lst in lda[bowCorpus]])
    return doc_topic_distro


dtd_test0=doc_topic_dist(df_test0)
dtd_test1=doc_topic_dist(df_test1)
dtd_test2=doc_topic_dist(df_test2)
dtd_test3=doc_topic_dist(df_test3)
dtd_test4=doc_topic_dist(df_test4)
dtd_test6=doc_topic_dist(df_test6)

#%%
# test1of2
# test of similarity
testing1vstraining1 = jsd_mat(dtd_test1, dtd_topic1)
# test of difference 
testing1vstraining4= jsd_mat(dtd_test1, dtd_topic4)
#%%
#test2of2
# semantic similarity test
testing4vTraining4 = jsd_mat(dtd_test4, dtd_topic4)
# semantic difference test
testing6vtraining4 = jsd_mat(dtd_test6, dtd_topic4)

#%%
js_plots([testing1vstraining1, testing1vstraining4], ['Virology (Testing vs Training)', 'Virology (Testing) vs Genetics (Training)'], color=['green','red'])

#%%

js_plots([testing4vTraining4, testing6vtraining4], ['Genetics (Testing vs Training)', 'Pathology (Testing) vs Genetics (Training)'])

#%%


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
        p2 = ['red', 'green']
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

js_plots_stacked([testing1vstraining1, testing1vstraining4],[testing4vTraining4, testing6vtraining4], ['Virology (Testing vs Training)', 'Virology (Testing) vs Genetics (Training)'], ['Genetics (Testing vs Training)', 'Pathology (Testing) vs Genetics (Training)'])

#%%
# similar query: Genetics vs 
msk = np.random.rand(len(df_test4)) < 0.80
ref_df = df_test4[msk]
ref_df.reset_index(drop=True, inplace=True)
query_df = df_test4[~msk]
query_df.reset_index(drop=True, inplace=True)
sim_query = doc_topic_dist(query_df)
sim_ref = doc_topic_dist(ref_df)
js_check1 = jsd_mat(sim_ref, sim_query)
js_ch1_mean = np.mean(js_check1, axis=0)
plt.hist(js_ch1_mean)
plt.xlabel('AJSD Means')
plt.ylabel('Frequency')
plt.title('Distribution of Means of Test set of Genetics Subset ')
plt.show()

#%%
 # 300x20 dtd
ref = jsd_mat(sim_ref, sim_ref)  # 300 x 300 ajsd
ref = ref[~np.eye(ref.shape[0], dtype=bool)].reshape(ref.shape[0], -1)  # remove ones => 250 by 250
means = np.mean(ref, axis=0)  # vector 250 means
sns.distplot(means, bins=30, color='purple')
plt.xlabel('AJSD Means')
plt.ylabel('Frequency')
plt.show()
#%%
prob_sim = []
for j in range(len(js_ch1_mean)):
    count1 = [i for i in means.flatten() if i <= js_ch1_mean[j]]  # at most inside means
    prob_sim.append(len(count1) / len(means))

print(len(count1))
# Probability that mean of new document is as high as our means for our reference set or lower
plt.hist(prob_sim, bins=20)

plt.title('Probabilities of Means of New Docs (Similar) inside generated distribution')
plt.ylabel('Frequency')
plt.xlabel('Probability')
plt.show()

plt.hist(prob_sim, bins=20, cumulative=True)

plt.title('Probabilities of Means of New Docs (Similar) inside generated distribution')
plt.ylabel('Frequency')
plt.xlabel('Cumulative Probability')
plt.show()

save_obj(prob_sim, 'sim_probs_applicationLDA')


#%%
#disimilar query: Genetics (ref) vs Virology (Disimi Query)

msk = np.random.rand(len(df_test1)) < 0.80

query_df = df_test1[~msk]
query_df.reset_index(drop=True, inplace=True)
disim_query = doc_topic_dist(query_df)

js_check2 = jsd_mat(sim_ref, disim_query)
js_ch2_mean = np.mean(js_check2, axis=0)
plt.hist(js_ch2_mean)
plt.xlabel('AJSD Means')
plt.ylabel('Frequency')
plt.title('Distribution of Means of Test set of Genetics Subset ')
plt.show()

#%%

prob_disim = []
for j in range(len(js_ch2_mean)):
    count1 = [i for i in means.flatten() if i <= js_ch2_mean[j]]  # at most inside means
    prob_disim.append(len(count1) / len(means))

print(len(count1))
# Probability that mean of new document is as high as our means for our reference set or lower
plt.hist(prob_disim, bins=20)

plt.title('Probabilities of Means of New Docs (Similar) inside generated distribution')
plt.ylabel('Frequency')
plt.xlabel('Probability')
plt.show()

plt.hist(prob_disim, bins=20, cumulative=True)

plt.title('Probabilities of Means of New Docs (Similar) inside generated distribution')
plt.ylabel('Frequency')
plt.xlabel('Cumulative Probability')
plt.show()

save_obj(prob_disim, 'disim_probs_application')