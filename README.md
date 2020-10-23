# Word-embeddings-for-semantic-similarity-comparing-LDA-with word2vec
Code for the comparing LDA and word2vec using the CORD-19 and 20 Newsgroups datasets <br>

## Methodology-20 Newsgroups
*The code contained in this folder are all the Python files necessary to carry out the experiments discussed in the Methodology.* <br>

**1. lda.py** - In this file , the entire corpus of the 20 Newsgroups dataset was extracted from the scikit-learn library and used to train the LDA model from gensim. Some basic text preprocessing was carried out before the model training. After the training of the model test sets were indexed and then compared amongst one another using the adjusted Jensen-Shannon distances (AJSDs) and histograms of the AJSDs were created. Once this is done a reference set and query set where generated and stored from the test sets and used to create the relevance index (the process of doing this is explained in the report). <br>
**2. w2v.py** - In this file, the entire corpus of the 20 Newsgroups dataset was extracted from the scikit-learn library and used to train the word2vec model from gensim. Once the model is trained, test sets were indexed and compared using the soft-cosine similarity. Finally a reference and query set was used to create and store the data needed to create the relevance index. <br> 
**3. rel_index.py** - Using the stored data from lda.py and w2v.py we create a relevance index and calculate some basic machine learning statistics.<br>


## application
*The code contained in this folder are Python files that carry out all the experiments in comparing LDA and word2vec as well as constructing the relevance index in application to the CORD-19 Dataset.* <br>

**1. application.py** - This file calls the cord-19-tools package so that we can obtain the full text articles. These are stored as pandas dataframes (df) and are then extracted for later use. <br>
**2. application-lda.py** - Basic text preprocessing, LDA model training using the gensim library, and obtaining of dominant topics for each document are carried out in this file. Furthermore, the df with the dominant topics are stored for later use with word2vec. Comparisons between topics using test sets are carried out using the AJSDs and data for the relevance index is extracted.  <br>
**3. w2v-application.py** - Using the dominant topics df as labels for which topic a doucment belonged to, we carried out some basic text preprocessing, trained the model using the gensim library and made comparisons between the topics. The data for the relevance index was calculated and stored. <br>
**4. rel_index_application.py** - Using the data extracted from the other files, we generate a relevance index for documents.
