from itertools import combinations
from rouge import Rouge
import numpy as np

def bestSummary(sent_list, ref_summary, c = 3):
    rouge = Rouge()
    best_summary = ' '.join(sent_list[0:3]);

    best_rouge = rouge.get_scores(best_summary, ref_summary)[0]['rouge-1']['f'];
    best_ind = [0,1,2];

    counter = 0;

    #essentially, if the article is too long, we only probe the first 20 sentences.
    combination_bound = min(9, len(sent_list));

    for c in range(2,5):
        sent_inds = list(range(combination_bound))
        for combo in combinations(sent_inds , c):
            summ_prop = ' '.join([sent_list[i] for i in combo]);
            scores = rouge.get_scores(summ_prop, ref_summary)[0];
            # print(scores)
            r1f = scores['rouge-1']['f']
            if(r1f > best_rouge):
                best_rouge = r1f;
                best_summary = summ_prop
                best_ind = combo;
            counter+=1;
            # if(counter>200):
            #     break;

    return best_summary, best_rouge, best_ind;

## what if we search the document for maximal sentences based on rouge...then combine the top 3.
def bestSummary_individual(sent_list, ref_summary, cmax = 10):
    rouge = Rouge()
    ind_rouge = [];
    for sent in sent_list:
        scores = rouge.get_scores(sent, ref_summary)[0];
        ind_rouge.append(scores['rouge-1']['f'])
    ind_rouge = np.array(ind_rouge)

    best_summary = '';
    best_rouge = 0; best_inds = [];
    lower_bound = 3;
    if(len(sent_list) < lower_bound):
        lower_bound = len(sent_list);
    if(cmax > len(sent_list)):
        cmax = len(sent_list);
    for summary_length in range(lower_bound,cmax+1):
        ind = np.argpartition(ind_rouge, -1*summary_length)[-1*summary_length:]
        #print(ind)
        summary_prop = ' '.join([sent_list[i] for i in ind]);
        scores = rouge.get_scores(summary_prop, ref_summary)[0];
        rf1 = scores['rouge-1']['f'];
        if(rf1 > best_rouge):
            best_rouge = rf1;
            best_summary = summary_prop
            best_inds = ind;
    return best_summary, best_rouge, best_inds;

def bestSummary_iterative(sent_list, ref_summary, cmax = 10):
    '''
    iteratively tries the best sentences.
    :param sent_list:
    :param ref_summary:
    :param cmax:
    :return:
    '''
    rouge = Rouge()

    best_summary = '';
    best_rouge = 0; best_inds = [];

    #termination condition: if we can't improve the score adding in the ith sentence
    cur_best = [];
    max_probe_length = 50;
    for i in range(1, cmax+1):
        best_ind = -1;
        for j in range(0, min(max_probe_length, len(sent_list))):
            new_sum = ' '.join(cur_best+[sent_list[j]]);
            score = rouge.get_scores(new_sum, ref_summary)[0]['rouge-1']['f'];
            if(score>best_rouge):
                best_rouge = score;
                best_ind = j;
        if(best_ind!= -1):
            cur_best.append(sent_list[best_ind]);
            best_inds.append(best_ind);
        else:
            break;
    #best_summary = ' '.join(cur_best);
    return cur_best, best_rouge, best_inds;
