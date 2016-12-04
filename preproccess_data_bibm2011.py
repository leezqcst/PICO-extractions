
# coding: utf-8

# In[107]:

from bs4 import BeautifulSoup
import os
import lxml
from random import shuffle


# In[ ]:




# # Title

# ### preprocessing

# In[3]:

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
                    print "BAD"
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




# In[41]:

def process_abstract_to_determine_good(abstract_path):
    soup = BeautifulSoup(open(abstract_path).read())
    pmid = soup.find("abstract")['id']

    sentences = list(soup.findAll("s"))
    sentences_part = soup.find_all(section="participants");
    
    full_text = str(soup.find("fulltext").string)
    full_text_nospace = full_text.replace(' ', '')
        
    for part_phrase in sentences_part:
        part_phrase = str(part_phrase.string)
        part_phrase_nospace = part_phrase.replace(' ', '')
        index = full_text_nospace.find(part_phrase_nospace)
        if index == -1:
#             print len(sentences_part)
#             print ""
            if (len(sentences_part) == 1):
                return True
#             num_indices = []
            
#             for digit in str.digits:
#                 num_indices.append(part_phrase_nospace.find(digit))
            
#             print part_phrase
#             print full_text
            
#             print ""
#             print "##############################################"
    return False


# In[42]:

def get_data_in_interval_output(start, end):
    f = open('./bibm2011corpus-master/abstracts_1.txt', 'r')
    abstract_list = f.readlines()
    abstract_list = [x.strip() for x in abstract_list]
    final_list = abstract_list[start:end]
    
    num_bad = 0
    
    good_abstract_list = []
    
    for abstract in abstract_list:
        bad = process_abstract_to_determine_good(abstract)
        if bad:
            num_bad += 1
        else:
            good_abstract_list.append(abstract)
            
        
    print "BAD: ", num_bad
    print good_abstract_list
    
    out_string = ''
    
    for ab in good_abstract_list:
        out_string = out_string + ab + '\n'
    
    
    f = open('./bibm2011corpus-master/abstracts_2.txt', 'w')
    f.write(out_string)
    f.close()
    print "\n\n\n"
    print len(good_abstract_list)
    


# In[43]:

a = 'hello my name is hansa'
print a[0:5]


# In[44]:

get_data_in_interval_output(0, 10)


# In[ ]:




# In[ ]:




# In[112]:

def process_abstract(abstract_path):
    soup = BeautifulSoup(open(abstract_path).read())
    pmid = soup.find("abstract")['id']

    sentences = list(soup.findAll("s"))
    sentences_part = soup.find_all(section="participants");
    
    full_text = str(soup.find("fulltext").string)
    full_text_nospace = full_text.replace(' ', '')
    
#     print "TEXT: "
#     print full_text
#     print " "
    
    f = open(abstract_path[:-4] + '_tokens.txt', 'r')
    text = f.read()
    f.close()
    
    
    word_array = text.split(' ')
    tag_array = ['None']*len(word_array)

    
    for part_phrase in sentences_part:
        part_phrase = str(part_phrase.string)
        part_phrase_nospace = part_phrase.replace(' ', '')
        index = full_text_nospace.find(part_phrase_nospace)
        
#         print "PART PHRASE:"
#         print part_phrase
#         print "  "
        
        part_phrase_list = part_phrase.split(' ')
#         print part_phrase_list
        
#         print "first part_phrase bit: ", part_phrase_list[0]

        if not (index == -1):
            in_phrase = False;
            for i in range(0, len(word_array)):
                word = word_array[i]
#                 print word
                if in_phrase:
#                     print "in phrase"
                    tag_array[i] = 'P'
                    if part_phrase_list[-1] == word:
                        in_phrase = False
                else:
#                     print "not in phrase"
                    if word == part_phrase_list[0]:
#                         print "yes they match"
                        
                        
                        search_string = ''.join(word_array[i:i+len(part_phrase_list)])
#                         print "search string: ", search_string
#                         print 'part_phrase_nospace: ', part_phrase_nospace
                        if (part_phrase_nospace in search_string) or (search_string in part_phrase_nospace):
                            in_phrase = True
                            tag_array[i] = 'P'
                            
    return word_array, tag_array
                    


# In[142]:

def get_data_in_interval(start, end):
    f = open('./bibm2011corpus-master/abstracts_2.txt', 'r')
    abstract_list = f.readlines()
    abstract_list = [x.strip() for x in abstract_list]
    final_list = abstract_list[start:end]
    
    word_array = []
    tag_array = []
    
    for abstract in final_list:
        [word_array_curr, tag_array_curr] = process_abstract(abstract)
        word_array.append(word_array_curr)
        tag_array.append(tag_array_curr)

    return [word_array, tag_array]
        


# In[126]:

def get_all_data():
    return get_data_in_interval(0, 135)



# In[143]:

def get_all_data_train():
    return get_data_in_interval(0, 95)


# In[144]:

def get_all_data_dev():
    return get_data_in_interval(95, 122)


# In[145]:

def get_all_data_test():
    return get_data_in_interval(122, 135)


# In[104]:

# [word_array, tag_array] = get_data_in_interval(0, 10)


# In[105]:

# for i in range(0, len(word_array)):
#     if tag_array[i] == 'P':
#         print word_array[i]


# In[ ]:



