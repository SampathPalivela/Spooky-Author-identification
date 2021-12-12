#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


train_csv = pd.read_csv("D:\\OneDrive\\Desktop\\projects\\spooky authentication\\spooky-author-identification\\train.csv")
train_csv.head()


# In[5]:


test_csv = pd.read_csv("D:\\OneDrive\\Desktop\\projects\\spooky authentication\\spooky-author-identification\\test.csv")
test_csv.head()


# In[6]:


test_csv.shape


# In[7]:


train_csv["author"].unique()


# In[8]:


sns.countplot("author",data=train_csv)


# ## Feature engineering

# ### ==> remove stop words

# In[9]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# In[10]:


sw = stopwords.words('english')


# function to remove stop words

# In[11]:


def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)


# In[12]:


train_csv['text'] = train_csv['text'].apply(stopwords)
train_csv.head(10)


# ### ==> Remove punctuations

# In[13]:


import string
punctuations = string.punctuation
punctuations


# function to remove punctuations

# In[14]:


def remove_punctuations(text):
    for i in text:
        if i in punctuations:
            text = text.replace(i,"")
    return text


# In[15]:


train_csv['text'] = train_csv['text'].apply(remove_punctuations)
train_csv.head()


# ### ==> Lemmatization

# In[16]:


from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()


# In[17]:


from nltk.corpus import wordnet

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# In[18]:


def lemmatize(text):
  
    text = [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in text.split()]
    return " ".join(text)


# In[19]:


train_csv['text']=train_csv['text'].apply(lemmatize)
train_csv.head()


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tfid_vectorizer = TfidfVectorizer("english")

tfid_vectorizer.fit(train_csv['text'])

dictionary = tfid_vectorizer.vocabulary_.items()  


# In[21]:


vocab = []
count = []
# iterate through each vocab and count append the value to designated lists
for key, value in dictionary:
    vocab.append(key)
    count.append(value)
# store the count in panadas dataframe with vocab as index
vocab_after_stem = pd.Series(count, index=vocab)
# sort the dataframe
vocab_after_stem = vocab_after_stem.sort_values(ascending=False)
vocab_after_stem['zuro']
# plot of the top vocab
top_vacab = vocab_after_stem.head(20)
top_vacab.plot(kind = 'barh', figsize=(8,10), xlim= (19040, 19080))


# ### ==> Vectorizing

# In[22]:


tfid_matrix = tfid_vectorizer.transform(train_csv['text'])
# collect the tfid matrix in numpy array
array = tfid_matrix.todense()


# In[23]:


df = pd.DataFrame(array)
df.head(10)


# In[24]:


df['output'] = train_csv['author']
df['id'] = train_csv['id']
df.head(10)


# In[25]:


x=df.drop(columns = ['output','id'],axis=1)
y = df['output']


# In[26]:


from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV


# In[27]:


alpha_list1 = np.linspace(0.006, 0.1, 20)
alpha_list1 = np.around(alpha_list1, decimals=4)
alpha_list1


# In[28]:


parameter_grid = [{"alpha":alpha_list1}]


# In[29]:


classifier1 = MultinomialNB()
# gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
gridsearch1 = GridSearchCV(classifier1,parameter_grid, scoring = 'neg_log_loss', cv = 4)
# fit the gridsearch
gridsearch1.fit(x, y)


# In[30]:


results1 = pd.DataFrame()
# collect alpha list
results1['alpha'] = gridsearch1.cv_results_['param_alpha'].data
# collect test scores
results1['neglogloss'] = gridsearch1.cv_results_['mean_test_score'].data


# In[31]:


plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.plot(results1['alpha'], -results1['neglogloss'])
plt.xlabel('alpha')
plt.ylabel('logloss')
plt.grid()


# In[32]:


print("Best parameter: ",gridsearch1.best_params_)


# In[33]:


print("Best score: ",gridsearch1.best_score_) 


# In[34]:


tfid_matrix = tfid_vectorizer.transform(test_csv['text'])
# collect the tfid matrix in numpy array
array = tfid_matrix.todense()


# In[35]:


xtest = pd.DataFrame(array)
xtest.head(10)


# In[36]:


mb = MultinomialNB(alpha=0.0208)
mb.fit(x,y)
predictions = mb.predict(xtest)


# In[38]:


predictions


# In[40]:


mb.score(x,y)


# In[ ]:




