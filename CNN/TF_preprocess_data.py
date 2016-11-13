
# coding: utf-8

# In[4]:

import sys
sys.path.insert(0, '../')
import os

os.chdir('..')
from genia_features_2 import abstracts2features


# In[ ]:

X,Y = abstracts2features(word_array[1:10],tag_array[1:10],(1,1),False, w2v_size=100)


# In[ ]:




# In[ ]:



