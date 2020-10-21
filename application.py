#%%

# cord-19-tools: package developed specifically for text analytics in Python with the CORD-19 dataset
# contains collections of approximately 25000 research articles
import sys
import cotools as co
import pickle
# function to save object as a pickle file
def save_obj(obj, name):
    """Save some object to a pickle file

    Args:
        obj (any): variable name for object to save
        name (str): string-name for  object to save
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# only run this once
# cotools.download()
#%%

data = co.Paperset('custom_license')
print(str(sys.getsizeof(data))+' bytes')
print(f"{len(data)} papers")
#%%
# Paperset lazily collects all the data from the pdf json files

# takes approx 10 mins to run so beware

data = co.Paperset("comm_use_subset")
alldata=data[:]

#%%
# obtain the unpreporcessed raw data to generate a training set
training_data = co.texts(alldata)
#%%
save_obj(training_data,"training_application")

#%%
# what if i train the models using the comm_use_subset and test the models using the non_comm_use dataset
# create a test set index with the Paperset class
data = co.Paperset("noncomm_use_subset")
alldata=data[:]

#%%
# collect text data given index created by Paperset class
test_data=co.texts(alldata)
save_obj(test_data, "test_data_application")
