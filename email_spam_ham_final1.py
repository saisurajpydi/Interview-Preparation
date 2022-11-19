#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

DATASETS_DIR = 'datasets'
MODELS_DIR = 'models'
TAR_DIR = os.path.join(DATASETS_DIR, 'tar')

SPAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'
EASY_HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2'
HARD_HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2'


# In[2]:


import tarfile
import shutil
from urllib.request import urlretrieve

def download_dataset(url):
    """download and unzip data from a url into the specified path"""
    
    # create directory if it doesn't exist
    if not os.path.isdir(TAR_DIR):
        os.makedirs(TAR_DIR)
    
    filename = url.rsplit('/', 1)[-1]
    tarpath = os.path.join(TAR_DIR, filename)
    
    # download the tar file if it doesn't exist
    try:
        tarfile.open(tarpath)
    except:
        urlretrieve(url, tarpath)
    
    with tarfile.open(tarpath) as tar:
        dirname = os.path.join(DATASETS_DIR, tar.getnames()[0])
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=DATASETS_DIR)
        
        cmds_path = os.path.join(dirname, 'cmds')
        if os.path.isfile(cmds_path):
            os.remove(cmds_path)
    
    return dirname


# In[3]:


spam_dir = download_dataset(SPAM_URL)
easy_ham_dir = download_dataset(EASY_HAM_URL)
hard_ham_dir = download_dataset(HARD_HAM_URL)


# In[4]:


import glob

def load_dataset(dirpath):
    
    files = []
    filepaths = glob.glob(dirpath + '/*')
    for path in filepaths:
        with open(path, 'rb') as f:
            byte_content = f.read()
            str_content = byte_content.decode('utf-8', errors='ignore')
            files.append(str_content)
    return files


# In[5]:




spam = load_dataset(spam_dir)
easy_ham = load_dataset(easy_ham_dir)
hard_ham = load_dataset(hard_ham_dir)


# In[6]:


import sklearn.utils
import numpy as np

# create the full dataset
X = spam + easy_ham + hard_ham
y = np.concatenate((np.ones(len(spam)), np.zeros(len(easy_ham) + len(hard_ham))))

# shuffle the dataset
X, y = sklearn.utils.shuffle(X, y, random_state=42)


# In[7]:


from sklearn.model_selection import train_test_split

# split the data into stratified training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=42)
# check dataset shapes
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))


# In[8]:


def remove_header(email):
    return email[email.index('\n\n'):]

import re 

def is_url(s):
    url = re.match("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
                     "[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", s)
    return url is not None

def convert_url_to_word(words):
    for i, word in enumerate(words):
        if is_url(word):
            words[i] = 'URL'
    return words


def convert_num_to_word(words):
    for i, word in enumerate(words):
        if word.isdigit():
            words[i] = ''
    return words

def remove_punctuation(email):
    new_email = ""
    for c in email:
        if c.isalnum() or c.isspace():
            new_email += c
    return new_email


# In[9]:


from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
import re

class CleanEmails(BaseEstimator, TransformerMixin):
    def __init__(self, no_header=True, to_lowercase=True, url_to_word=True, num_to_word=True,
                 remove_punc=True,stop_words=True,stops=True):
        self.no_header = no_header
        self.to_lowercase = to_lowercase
        self.url_to_word = url_to_word
        self.num_to_word = num_to_word
        self.remove_punc = remove_punc
        self.stop_words=stop_words
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        XY_cleaned = []
        for email in X:
            if self.no_header:
                email = remove_header(email)
            if self.stop_words:
                comp=re.compile('<[^>]+>')
                phrase=re.sub(comp,"",email)
                phrase = re.sub(r"won't", "will not", phrase)
                phrase = re.sub(r"can\'t", "can not", phrase)
                phrase=re.sub(r"n\'t","not",phrase)
                phrase=re.sub(r"\'re","are",phrase)
                phrase = re.sub(r"\'s", "is", phrase)
                phrase = re.sub(r"\'d", "would", phrase)
                phrase = re.sub(r"\'ll", "will", phrase)
                phrase = re.sub(r"\'t", "not", phrase)
                phrase = re.sub(r"\'ve", "have", phrase)
                phrase = re.sub(r"\'m", "am", phrase)
                phrase=phrase.replace('\\r',"")
                phrase=phrase.replace('\\t',"")
                phrase=phrase.replace('\\n',"")
                phrase=phrase.replace('nan',"")
                email=phrase
            if self.to_lowercase:
                email = email.lower()
            
            email_words = email.split()
            if self.url_to_word:
                email_words = convert_url_to_word(email_words)
                
            if self.num_to_word:
                email_words = convert_num_to_word(email_words)
            email = ' '.join(email_words)
            
            if self.remove_punc:
                email = remove_punctuation(email)
            
            email_words=[]
            for i in email.split():
                if len(str(i))>3  and not i.isnumeric():
                    email_words.append(i)
            email=' '.join(email_words)
            
            
            
            XY_cleaned.append(email)
        return XY_cleaned


# In[10]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# full preparation pipeline
prepare_pipeline1 = Pipeline([
    ('clean_email', CleanEmails())
])


# In[ ]:





# In[11]:


X_train_prepared = prepare_pipeline1.fit_transform(X_train)

X_test_prepared = prepare_pipeline1.transform(X_test)


# In[12]:


print(X_train[0])
print(X_train_prepared[0])


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

vectorizer = CountVectorizer(stop_words = 'english')
data = vectorizer.fit_transform(X_train_prepared)

clustering_model = KMeans(n_clusters = 2, 
                          init = 'k-means++',
                          max_iter = 300, n_init = 10)
clustering_model.fit(data)
def accuracy(prediction,y_test):
    return f1_score(prediction, y_test, average='weighted')


# In[14]:


y=vectorizer.transform(X_test_prepared)
prediction = clustering_model.predict(y)
print(data.shape,y.shape)


# In[15]:


print(accuracy(prediction,y_test))


# In[16]:


prepare_pipeline = Pipeline([
    ('clean_email', CleanEmails()),
    ('bag_of_words', CountVectorizer())
])

X_train_prepared = prepare_pipeline.fit_transform(X_train)

from sklearn.metrics import precision_score, recall_score, f1_score

"""def eval_confusion(y_pred, y_true=y_train):
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'matrix': conf_matrix, 'precision': precision, 'recall': recall, 'f1': f1}"""


# In[17]:


import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(hidden_layer_sizes=(16,))
forest_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier()
classifiers = {
    'Random Forest': forest_clf,
    'KNN': knn_clf,
    'MLP': mlp_clf
}


# In[18]:


from sklearn.model_selection import cross_val_predict

# make predictions using each model
y_preds = {}
for clf_name, clf in classifiers.items():
    y_preds[clf_name] = cross_val_predict(clf, X_train_prepared, y_train, cv=3)

from sklearn.metrics import accuracy_score

# evaluate each classifier's accuracy
for clf_name, y_pred in y_preds.items():
    print("{}:".format(clf_name))
    print(accuracy_score(y_train, y_pred))
    print()


# In[19]:


from sklearn.cluster import OPTICS
import numpy as np

X_train_prepared = prepare_pipeline1.fit_transform(X_train)

vectorizer = CountVectorizer(stop_words = 'english')
data = vectorizer.fit_transform(X_train_prepared)
y=vectorizer.transform(X_test_prepared)
pradition = clustering_model.predict(y)
clustering=OPTICS(min_samples=2).fit(data.toarray())
clustering.fit(data)

print(accuracy(prediction,y_test))


# In[19]:


from sklearn.cluster import DBSCAN

X_train_prepared = prepare_pipeline1.fit_transform(X_train)

vectorizer = CountVectorizer(stop_words = 'english')
data = vectorizer.fit_transform(X_train_prepared)
y=vectorizer.transform(X_test_prepared)
pradiction = clustering_model.predict(y)

clustering = DBSCAN(eps=3, min_samples=2).fit(data)
print(accuracy(prediction,y_test))


# In[ ]:





# In[ ]:





# In[ ]:




