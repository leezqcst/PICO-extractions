
# coding: utf-8

# In[78]:

from bs4 import BeautifulSoup
import os
import lxml
from random import shuffle


# In[ ]:




# In[ ]:




# In[147]:

def extract_and_write_good_abstracts():
    directory = 'bibm2011corpus-master'

    good_abstracts_list = []

    # For each subdirectory
    for subdir in os.listdir(directory):
        subdir_path = directory + '/' + subdir
        ann_subdir_path = directory + '/' + subdir
        # print subdir_path

        # Not a directory
        if not os.path.isdir(subdir_path):
            continue

        # For each abstract in subdirectory
        for abstract in os.listdir(subdir_path):
            if (abstract.endswith('.xml')):
                abstract_path = subdir_path + '/' + abstract; 
                
                full_text_txt_path = abstract_path[:-4] + '.txt';
                f = open(full_text_txt_path, 'w')
                
                soup = BeautifulSoup(open(abstract_path).read())
                pmid = soup.find("abstract")['id']

                sentences = list(soup.findAll("s"))
                sentances_part = soup.find_all(section="participants");
                fulltext = soup.find('fulltext')
                if (fulltext.string == None):
                    print "DAMN"
                else:
                    out = str(fulltext.string)
                    f.write(out)
                
                f.close()
                
                use_abstract = True;

                for i in sentances_part:
                    if i.string == None:
                        use_abstract = False;

                if use_abstract:
                    good_abstracts_list.append(abstract_path)
#                     good_abstracts_list = good_abstracts_list + abstract_path + '\n'
    
#     shuffle(good_abstracts_list)
    
#     output_list = ''
#     for item in good_abstracts_list:
#         output_list = output_list + item + '\n'
    
#     f = open('./bibm2011corpus-master/abstracts_1.txt', 'w')
#     f.write(output_list)
#     f.close()


# In[148]:

extract_and_write_good_abstracts()


# In[ ]:




# In[90]:

f = open('./bibm2011corpus-master/abstracts_1.txt', 'r')
abstract_list = f.readlines()


# In[91]:

abstract_list = [x.strip() for x in abstract_list]


# In[ ]:




# In[ ]:




# In[139]:

def process_abstract(abstract_path):
    soup = BeautifulSoup(open(abstract_path).read())
    pmid = soup.find("abstract")['id']

    sentences = list(soup.findAll("s"))
    sentances_part = soup.find_all(section="participants");
    
    full_text = str(soup.find("fulltext").string)
    
    participant_phrase = str(sentances_part[0].string)
    
    
    
    print "original part phrase: ", participant_phrase
    print ""
    print str(sentances_part[1].string)
    index = full_text.find(participant_phrase)
    print "INDEX: ", index
    print full_text[index:(index+len(participant_phrase))]
    print ""
    index2 = full_text.find(str(sentances_part[1].string))
    print "INDEX2: ", index2
    print full_text[index2:(index2+(len(str(sentances_part[1].string))))]
    print ""
    print full_text


# In[140]:

def get_data_in_interval(start, end):
    f = open('./bibm2011corpus-master/abstracts_1.txt', 'r')
    abstract_list = f.readlines()
    abstract_list = [x.strip() for x in abstract_list]
    final_list = abstract_list[start:end]
    
    process_abstract(abstract_list[0])
    


# In[141]:

a = 'hello my name is hansa'
print a[0:5]


# In[142]:

get_data_in_interval(0, 10)


# In[ ]:




# In[47]:

def get_all_data()
for abstract_path in abstract_list:
    soup = BeautifulSoup(open(abstract_path).read())
    pmid = soup.find("abstract")['id']

    sentences = list(soup.findAll("s"))
    sentances_part = soup.find_all(section="participants");

    for i in sentances_part:
        if i.string == None:
            print "FUCK"


# In[ ]:



