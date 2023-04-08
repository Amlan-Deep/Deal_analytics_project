#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyLDAvis')


# In[2]:


import pandas as pd
import numpy as np

import re
import string

import spacy

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from google.colab import drive
drive.mount('/content/drive/')


# In[4]:


with open("/content/drive/MyDrive/Deal_analytics_project/Amendment Contract.txt") as file:amendment_contract = file.read()
amendment_contract


# In[5]:


with open("/content/drive/MyDrive/Deal_analytics_project/board resolution.txt") as file:board_text = file.read()
with open("/content/drive/MyDrive/Deal_analytics_project/Software License Agreement Contract.txt") as file:software_license_agreement = file.read()
with open("/content/drive/MyDrive/Deal_analytics_project/tenant lease.txt") as file:tenant_lease = file.read()


# In[6]:


import nltk
nltk.download('punkt')
nltk.download('omw-1.4')


# In[7]:


from nltk import sent_tokenize


# In[8]:


list_of_sentence = sent_tokenize(amendment_contract)
list_of_sentence1 = sent_tokenize(software_license_agreement)
list_of_sentence2 = sent_tokenize(board_text)
list_of_sentence3 = sent_tokenize(tenant_lease)


# In[9]:


import nltk
nltk.download('stopwords')


# In[10]:


list_of_sentence


# In[11]:


list_of_sentence1


# In[12]:


get_ipython().system('pip install import-ipynb')

from google.colab import drive
drive.mount('/content/drive')
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Deal_analytics_project/')


# In[13]:


import web_scraping_module


# In[14]:


from web_scraping_module import newsletter
from web_scraping_module import investmentletter


# In[15]:


newsletter()


# In[16]:


investmentletter()


# In[17]:


import nltk
newslist=[]
newslist = web_scraping_module.newsletter()


# In[18]:


import re
from nltk.tokenize import word_tokenize

new_newsletter = []

for sentence in newslist:
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove words that contain digits
    filtered_words = [word for word in words if not re.search('\d', word)]
    # Append the filtered words to the list of new sentences
    new_newsletter.append(filtered_words)

new_newsletter



# In[19]:


list_of_simple_preprocess_data = []
for i in list_of_sentence:
    list_of_simple_preprocess_data.append(gensim.utils.simple_preprocess(i, deacc=True, min_len=3))

list_of_simple_preprocess_data1 = []
for i in list_of_sentence1:
    list_of_simple_preprocess_data1.append(gensim.utils.simple_preprocess(i, deacc=True, min_len=3))

list_of_simple_preprocess_data2 = []
for i in list_of_sentence2:
    list_of_simple_preprocess_data2.append(gensim.utils.simple_preprocess(i, deacc=True, min_len=3))

list_of_simple_preprocess_data3 = []
for i in list_of_sentence3:
    list_of_simple_preprocess_data3.append(gensim.utils.simple_preprocess(i, deacc=True, min_len=3))



# In[20]:


texts = list_of_simple_preprocess_data
texts1 = list_of_simple_preprocess_data1
texts2 = list_of_simple_preprocess_data2
texts3 = list_of_simple_preprocess_data3


# In[21]:


texts


# In[22]:


texts1


# In[23]:


bigram = gensim.models.Phrases(list_of_simple_preprocess_data) 


# In[24]:


bigram


# In[25]:


from nltk.corpus import stopwords


# In[26]:


stops = set(stopwords.words('english')) 
stops


# In[27]:


import re
get_ipython().system('pip install Pattern library')


# In[28]:


texts = [[word for word in line if word not in stops] for line in texts]
texts1 = [[word for word in line if word not in stops] for line in texts1]
texts2 = [[word for word in line if word not in stops] for line in texts2]
texts3 = [[word for word in line if word not in stops] for line in texts3]
new_newsletter=[[word for word in line if word not in stops] for line in new_newsletter]
texts_final = []
texts_final.extend(texts)
texts_final.extend(texts1)
texts_final.extend(texts2)
texts_final.extend(texts3)
texts_final.extend(new_newsletter)
texts_final


# In[29]:


from gensim.models import LdaModel
from gensim.corpora import Dictionary


# In[30]:


dictionary=Dictionary(texts_final)
corpus = [dictionary.doc2bow(text) for text in texts_final]


# In[31]:


print(dictionary)


# In[32]:


print(corpus)


# In[33]:


ldamodel = LdaModel(corpus=corpus, num_topics=7, id2word=dictionary)


# In[34]:


ldamodel.show_topics()


# In[35]:


import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()


# In[36]:


vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
vis


# In[36]:




