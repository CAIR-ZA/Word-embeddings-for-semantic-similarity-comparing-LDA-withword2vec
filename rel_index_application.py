# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:24:27 2020

@author: Tristan Harris
"""

#%%

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#%%
# LDA
sim_prob = load_obj('sim_probs_applicationLDA')
disim_prob = load_obj('disim_probs_application')

disim_prob = pd.DataFrame(disim_prob)

# disim_prob = np.random.choice(disim_prob[0], size=10000, replace=False)

sim_prob = pd.DataFrame(sim_prob)


# sim_prob = np.random.choice(sim_prob[0], size=10000, replace=False)
unrel_labels = ['r'] * len(disim_prob)

rel_labels = ['b'] * len(sim_prob)

labels = np.concatenate((unrel_labels, rel_labels))
jsd = np.concatenate((disim_prob, sim_prob))
# create dataframe of rel_index and label
df_original = pd.DataFrame(data=[labels, jsd]).T
df_original['labels'] = df_original[0]
df_original['rel_index'] = df_original[1]



df = df_original.sort_values('rel_index', ascending=False)

# find threshold (minimum relevance index of relevant articles)
# threshold = df.loc[df['labels'] == 'b']['rel_index'].min()
threshold = 0.10
# true relevant (relevant articles above threshold)
true_relevant = df.loc[(df["labels"] == 'b') & (df["rel_index"] >= threshold)]
# false relevant (relevant articles below the threshold)
false_relevant = df.loc[(df["labels"] == 'b') & (df["rel_index"] < threshold)]

# true irrelevant (irrelevant articles below threshold)
true_irrelevantlda = df.loc[(df["labels"] == 'r') & (df["rel_index"] < threshold)]
# false irrelevant (irrelevant articles above threshold)
false_irrelevant = df.loc[(df["labels"] == 'r') & (df["rel_index"] >= threshold)]
confusion_matrix = np.mat([[len(true_relevant), len(true_irrelevantlda), sum((len(true_relevant), len(true_irrelevantlda)))],
                           [len(false_relevant), len(false_irrelevant),
                            sum((len(false_irrelevant), len(false_relevant)))],
                           [sum((len(true_relevant), len(false_relevant))),
                            sum((len(false_irrelevant), len(true_irrelevantlda))), len(df)]])
conf_df = pd.DataFrame(confusion_matrix
                       , columns=["Relevant", "Irrelevant", "Total"]
                       , index=["True", "False", "Total"])
# find threshold (minimum relevance index of relevant articles)
# threshold = df.loc[df['labels'] == 'b']['rel_index'].min().astype(float)

perc_ignorelda = float(len(true_irrelevantlda)) / (len(df)) * 100  # percentage of documents to ignore
precision = float(len(true_irrelevantlda)) / float(len(false_irrelevant) + len(true_irrelevantlda))
accuracy = (len(true_relevant) + len(true_irrelevantlda)) / len(df)
recall = float(len(true_irrelevantlda)) / float(len(false_relevant) + len(true_irrelevantlda))

fig = plt.figure(figsize=(10, 7.5), tight_layout=True)

lda_df = df
lda_df.reset_index(drop=True, inplace=True)

plt.ylim([0, 1])
plt.xlim([0, len(labels)])
plt.scatter(range(len(df)), df['rel_index'], c=df['labels'], s=30, alpha=0.7)
# plot threshold
plt.axhline(threshold, c='green', linewidth=1.5)
plt.ylabel('Relevance Index')
plt.xlabel('Document Count')
plt.title('LDA Similarity Test: Relevance Index ')
plt.savefig('LDA_RelIndexMeans.jpg', bbox_inches='tight')
plt.show()

print('* LDA Evaluation *\n')
print("Threshold:", threshold)
print(r'True irrelevant: ' + str(len(true_irrelevantlda)) + '\n' + 'From total of: ' + str(
    len(df)) + '  (' + "%.2f" % perc_ignorelda + '%)')
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy: " + str(accuracy) + '\n')

# W2V
#%%
sim_prob = load_obj('probsim_w2v-application')
disim_prob = load_obj('probdisim_w2v-application')

disim_prob = pd.DataFrame(disim_prob)

sim_prob = pd.DataFrame(sim_prob)

# sim_prob = np.random.choice(sim_prob[0], size=10000, replace=False)
unrel_labels = ['r'] * len(disim_prob)

rel_labels = ['b'] * len(sim_prob)

labels = np.concatenate((unrel_labels, rel_labels))
jsd = np.concatenate((disim_prob, sim_prob))
# create dataframe of rel_index and label
df_original = pd.DataFrame(data=[labels, jsd]).T
df_original['labels'] = df_original[0]
df_original['rel_index'] = df_original[1]

df = df_original.sort_values('rel_index', ascending=False)

# find threshold (minimum relevance index of relevant articles)
# threshold = df.loc[df['labels'] == 'b']['rel_index'].min()
threshold = 0.10
# true relevant (relevant articles above threshold)
true_relevant = df.loc[(df["labels"] == 'b') & (df["rel_index"] >= threshold)]
# false relevant (relevant articles below the threshold)
false_relevant = df.loc[(df["labels"] == 'b') & (df["rel_index"] < threshold)]

# true irrelevant (irrelevant articles below threshold)
true_irrelevant = df.loc[(df["labels"] == 'r') & (df["rel_index"] < threshold)]
# false irrelevant (irrelevant articles above threshold)
false_irrelevant = df.loc[(df["labels"] == 'r') & (df["rel_index"] >= threshold)]
confusion_matrix = np.mat([[len(true_relevant), len(true_irrelevant), sum((len(true_relevant), len(true_irrelevant)))],
                           [len(false_relevant), len(false_irrelevant),
                            sum((len(false_irrelevant), len(false_relevant)))],
                           [sum((len(true_relevant), len(false_relevant))),
                            sum((len(false_irrelevant), len(true_irrelevant))), len(df)]])
conf_df = pd.DataFrame(confusion_matrix
                       , columns=["Relevant", "Irrelevant", "Total"]
                       , index=["True", "False", "Total"])
# find threshold (minimum relevance index of relevant articles)
# threshold = df.loc[df['labels'] == 'b']['rel_index'].min().astype(float)

perc_ignore = float(len(true_irrelevant)) / (len(df)) * 100  # percentage of documents to ignore
precision = float(len(true_irrelevant)) / float(len(false_irrelevant) + len(true_irrelevant))
accuracy = (len(true_relevant) + len(true_irrelevant)) / len(df)
recall = float(len(true_irrelevant)) / float(len(false_relevant) + len(true_irrelevant))

w2v_df = df
w2v_df.reset_index(drop=True, inplace=True)

fig = plt.figure(figsize=(10, 7.5), tight_layout=True)

plt.ylim([0, 1])
plt.xlim([0, len(labels)])
plt.scatter(range(len(df)), df['rel_index'], c=df['labels'], s=30, alpha=0.7)
# plot threshold
plt.axhline(threshold, c='green', linewidth=1.5)
plt.ylabel('Relevance Index')
plt.xlabel('Document Count')
plt.title('W2V Similarity Test: Relevance Index')
plt.savefig('w2v_RelIndexMeans.jpg', bbox_inches='tight')
plt.show()

print('* W2V Evaluation *\n')
print("Threshold:", threshold)
print(r'True irrelevant: ' + str(len(true_irrelevant)) + '\n' + 'From total of: ' + str(
    len(df)) + '  (' + "%.2f" % perc_ignore + '%)')
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy: " + str(accuracy) + '\n')


# plotting subplots
fig, axs = plt.subplots(2,1, figsize=(12.5, 10), sharex=True)
[axs[i].set_ylim([0, 1]) for i in range(2)]
[axs[j].set_xlim([0, len(labels)]) for j in range(2)]
axs[0].scatter(range(len(lda_df)), lda_df['rel_index'], c=lda_df['labels'], s=30, alpha=0.7)
axs[1].scatter(range(len(w2v_df)), w2v_df['rel_index'], c=w2v_df['labels'], s=30, alpha=0.7)


axs[0].text(x=145, y=0.15,s=r'Threshold = 0.1'+'\n'+r'True irrelevant = ' + str(len(true_irrelevantlda)) + '\n' + 'From '
                                                                                                              'total '
                                                                                                              'of: '
                           + str(len(lda_df)) + ' docs (' + "%.2f" % perc_ignorelda + '%)')

axs[1].text(x=45, y=0.15, s=r'Threshold = 0.1'+'\n'+r'True irrelevant = ' + str(len(true_irrelevant)) + '\n' + 'From '
                                                                                                            'total '
                                                                                                            'of: ' +
                            str(len(w2v_df)) + ' docs (' + "%.2f" % perc_ignore + '%)')
# plot threshold
[axs[i].axhline(threshold, c='green', linewidth=1.5) for i in range(2)]
[axs[i].set_ylabel('Relevance Index (Probabilities)') for i in range(2)]
axs[1].set_xlabel('Document Count')
axs[0].set_title('LDA Relevance Index: Genetics Ref-Query (Blue) vs Genetics-Virology (Red)', fontsize='large')
axs[1].set_title('Word2vec Relevance Index: Genetics Ref-Query (Blue) vs Genetics-Virology (Red)', fontsize='large')
plt.savefig('w2vVSlda_relindex_chrvwin.jpg', bbox_inches='tight')
plt.show()

# w2v very sensitive to changes in threshold where lda is robust
