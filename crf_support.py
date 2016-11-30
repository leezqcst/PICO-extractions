
# coding: utf-8

# ## CRF Support
# 

# In[ ]:

from collections import defaultdict


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


# In[ ]:

def compare_tags(pred_tags_list, gold_tags_list, eval_tag):
    count_pred = defaultdict(int)
    length_pred = defaultdict(int)
    count_gold = defaultdict(int)
    length_gold = defaultdict(int)
        
    for i in range(len(pred_tags_list)):
        pred_tags = pred_tags_list[i]
        gold_tags = gold_tags_list[i]

        def to_intervals(tags):
            intervals = []
            curr_start = None

            for j in range(len(tags)+1):
                if j == len(tags):
                    if curr_start != None:
                        intervals.append((curr_start, j))
                        curr_start = None
                elif tags[j] == eval_tag:
                    if curr_start == None:
                        curr_start = j
                else:
                    if curr_start != None:
                        intervals.append((curr_start, j))
                        curr_start = None

            return intervals
    
        pred_intervals = to_intervals(pred_tags)
        gold_intervals = to_intervals(gold_tags)
        
        def evaluate_intervals(source_intervals, target_intervals, count_dict, length_dict):
            def relationship(interval_1, interval_2):
                a, b = interval_1
                c, d = interval_2

                if a == c and b == d:
                    return 'Identical'
                elif b <= c or d <= a:
                    return 'Non-overlapping'
                elif c <= a and b <= d:
                    return 'Subinterval'
                elif a <= c and d <= b:
                    return 'Superinterval'
                else:
                    return 'Overlapping'

            def one_to_many_relationship(interval, intervals):
                encountered = set()
                for target_interval in intervals:
                    relation = relationship(interval, target_interval)
                    encountered.add(relation)

                for relation in ['Identical', 'Subinterval', 'Superinterval', 'Overlapping']:
                    if relation in encountered:
                        return relation

                return 'Non-overlapping'
            
            for interval in source_intervals:
                
                relation = one_to_many_relationship(interval, target_intervals)
                
                a, b = interval
                
                count_dict[relation] += 1
                length_dict[relation] += b-a
            
            return
        
        evaluate_intervals(pred_intervals, gold_intervals, count_pred, length_pred)
        evaluate_intervals(gold_intervals, pred_intervals, count_gold, length_gold)
    
    def print_result(count_pred, length_pred, count_gold, length_gold):
        types = ['Identical', 'Subinterval', 'Superinterval', 'Overlapping', 'Non-overlapping']
        
        switcher = [
            (count_pred, 'predicted', 'intervals'),
            (length_pred, 'predicted', 'tokens'),
            (count_gold, 'gold', 'intervals'),
            (length_gold, 'gold', 'tokens')
        ]
        
        for dictionary, p_or_g, i_or_t in switcher:
            total = sum(dictionary.values())
            
            print 'There are {} {} {}:'.format(total, p_or_g, i_or_t)
            for interval_type in types:
                print 'Number of type {}: {}'.format(interval_type+' '*(15-len(interval_type)),                                                     dictionary[interval_type])
            print ''
            
        return
    
    print_result(count_pred, length_pred, count_gold, length_gold)
    return


# In[ ]:

def filter_phrase(tags_list, genia_tags_list, phrase='NP'):
    filtered_tags_list = []
    
    for i in range(len(tags_list)):
        tags = tags_list[i]
        genia_tags = genia_tags_list[i]
        
        phrases = [x[3][2:] for x in genia_tags]
        
        filtered_tags = [tags[j] for j in range(len(tags)) if phrases[j] == phrase]
        
        filtered_tags_list.append(filtered_tags)
    
    return filtered_tags_list

