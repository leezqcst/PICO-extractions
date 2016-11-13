
# coding: utf-8

# In[1]:

import os


# In[45]:

abstract = 'PICO-annotations/batch5k/0074f5e102cf4409ac07f6209dd30144/20957980.txt'
abs_file = open(abstract, 'r');
text = abs_file.read()

ann = [24, 65, 322, 375, 470 ,517,529, 695, 697, 808, 900, 1020, 1330, 1410 ]


# In[46]:

text


# In[47]:

text_clean = text_clean.replace('\n', '')
text_clean = text_clean.replace(' ', '')
text_clean


# In[48]:

text


# In[56]:

new_index = 0

ann_ind = 0
curr_ann = ann[ann_ind]

new_anns = []

for c in range(len(text)):
    if text[c] == ' ' or text[c] == '\n':
        new_index -= 1
    if c == curr_ann:
        new_anns.append(new_index)
        ann_ind += 1
        if (ann_ind >= len(ann)):
            break
        curr_ann = ann[ann_ind]
    new_index += 1


# In[57]:

ann


# In[58]:

new_anns


# In[59]:

for i in range(0, len(ann), 2):
    print text[ann[i]:ann[i+1]]
    print text_clean[new_anns[i]:new_anns[i+1]]
    print " "


# In[ ]:



