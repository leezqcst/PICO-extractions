
# coding: utf-8

# ## CRF Support
# 

# In[ ]:

def get_all_data(data_set, tag):
    switcher = {
        'train': 'PICO-annotations/train_abstracts.txt',
        'dev': 'PICO-annotations/dev_abstracts.txt',
        'test': 'PICO-annotations/test_abstracts.txt', 
    }
    
    path = switcher[data_set]
    abstract_file = open(path, 'r')
    abstracts = abstract_file.readlines()
    abstract_file.close()
    
    abstracts = [x.strip() for x in abstracts]
    
    tokens_array = []
    tags_array = []
    
    for abstract_path in abstracts:
        token_path = '{}_tokens.txt'.format(abstract_path[:-4])
        tag_path = '{}_{}_tokens_tags.ann'.format(abstract_path[:-4], tag)
        
        f = open(token_path, 'r')
        tokens = f.read().split()
        f.close()
        
        f = open(tag_path, 'r')
        tags = f.read().split()
        f.close()
        
        if len(tokens) != len(tags):
            raise ValueError('For this file, len of abstract words and tags did not match.', abstract_path)
        
        tokens_array.append(tokens)
        tags_array.append(tags)
    
    if len(tokens_array) != len(tags_array):
        raise ValueError('Overall, len of abstract words and tags did not match.')
    
    return tokens_array, tags_array

