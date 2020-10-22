import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


sim_prob = load_obj('probsim_Similar_mean2ref')
disim_prob = load_obj('probsim_Disim_meantoref')
disim_means = load_obj('disim_means')
sim_means = load_obj('sim_means')
disim_prob = pd.DataFrame(disim_prob)
disim_means = pd.DataFrame(disim_means)
# disim_prob = np.random.choice(disim_prob[0], size=10000, replace=False)

sim_prob = pd.DataFrame(sim_prob)
sim_means = pd.DataFrame(sim_means)

# sim_prob = np.random.choice(sim_prob[0], size=10000, replace=False)
unrel_labels = ['r'] * len(disim_prob)

rel_labels = ['b'] * len(sim_prob)

labels = np.concatenate((unrel_labels, rel_labels))
jsd = np.concatenate((disim_prob, sim_prob))
means = np.concatenate((disim_means, sim_means))
# create dataframe of rel_index and label
df_original = pd.DataFrame(data=[labels, jsd, means]).T
df_original['labels'] = df_original[0]
df_original['rel_index'] = df_original[1]
df_original['means'] = df_original[2]

plt.ylim([0, 1])
plt.xlim([min(means), max(means)])
plt.scatter(df_original['means'], df_original['rel_index'], c=df_original['labels'])
# plot threshold
# plt.axhline(threshold, c='green', linewidth=1.5)
plt.ylabel('Relevance index')
plt.xlabel('AJSD Means')
plt.title('LDA Similarity Test: Probability-Means Scatter Plot ')
plt.savefig('LDA_RelIndex_meansto_probSPlot.jpg', bbox_inches='tight')
plt.show()

df = df_original.sort_values('rel_index', ascending=False)

# find threshold (minimum relevance index of relevant articles)
# threshold = df.loc[df['labels'] == 'b']['rel_index'].min()
threshold = 0.1
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

fig = plt.figure(figsize=(10, 7.5), tight_layout=True)

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
print(r'True irrelevant: ' + str(len(true_irrelevant)) + '\n' + 'From total of: ' + str(
    len(df)) + '  (' + "%.2f" % perc_ignore + '%)')
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy: " + str(accuracy) + '\n')

# W2V

sim_prob = load_obj('probsim_Similar_mean2refw2v')
disim_prob = load_obj('probsim_Disim_meantorefw2v')
disim_means = load_obj('w2vmeans_disim')
sim_means = load_obj('w2vmeans_sim')
disim_prob = pd.DataFrame(disim_prob)
disim_means = pd.DataFrame(disim_means)
# disim_prob = np.random.choice(disim_prob[0], size=10000, replace=False)

sim_prob = pd.DataFrame(sim_prob)
sim_means = pd.DataFrame(sim_means)

# sim_prob = np.random.choice(sim_prob[0], size=10000, replace=False)
unrel_labels = ['r'] * len(disim_prob)

rel_labels = ['b'] * len(sim_prob)

labels = np.concatenate((unrel_labels, rel_labels))
jsd = np.concatenate((disim_prob, sim_prob))
means = np.concatenate((disim_means, sim_means))
# create dataframe of rel_index and label
df_original = pd.DataFrame(data=[labels, jsd, means]).T
df_original['labels'] = df_original[0]
df_original['rel_index'] = df_original[1]
df_original['means'] = df_original[2]

plt.ylim([0, 1])
plt.xlim([min(means), max(means)])
plt.scatter(df_original['means'], df_original['rel_index'], c=labels, s=30, alpha=0.7)
# plot threshold
# plt.axhline(threshold, c='green', linewidth=1.5)
plt.ylabel('Relevance index')
plt.xlabel('SCS Means')
plt.title('W2V Similarity Test: Probability-SCS Means Scatter Plot ')
plt.savefig('w2v_RelIndex_meansto_probSPlot.jpg', bbox_inches='tight')
plt.show()

df = df_original.sort_values('rel_index', ascending=False)

# find threshold (minimum relevance index of relevant articles)
# threshold = df.loc[df['labels'] == 'b']['rel_index'].min()
threshold = 0.1
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

# w2v very sensitive to changes in threshold where lda is robust


