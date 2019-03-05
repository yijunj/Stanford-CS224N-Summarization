import sys
from rouge import Rouge
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from artifical_summary.sentence_combination import *
from collections import Counter

sys.path.append("D:\Documents\Classes\CS224n\project")

rouge = Rouge()

## cycle through articles

data = "../processed_train/newsroom_train_99844_clean.p";

(summaries, texts) = pickle.load(open(data, 'rb'));
print(len(texts))
print(len(summaries))

rouge_data = [];
sentence_lengths = [];
num_sents_summary = 3;
enhanced_rouge = [];
ind_distribution = [];
extractive_summaries = [];
preserved_texts = [];
preserved_summaries = [];
for c in range(len(summaries)):
    # if(c > 200):
    #     break;
    if (c % 100 == 1):
        print('processed %d samples' % c)
        print(np.mean(enhanced_rouge))
    # if((c+1)%25000 == 0):
    #     pickle.dump((preserved_summaries, preserved_texts, extractive_summaries),
    #                 open('algorithmic_extraction_summaries_%d.p' % len(extractive_summaries), 'wb'))

    text = texts[c];
    summary = summaries[c];
    if (len(text) <= 1):
        continue;
    first3Sentences = ' '.join(text[0:num_sents_summary])
    # print(first3Sentences)
    sentence_lengths.append(len(text))
    #try:
    scores = rouge.get_scores(first3Sentences, summary)[0];
    # print(scores)
    r1f = scores['rouge-1']['f']
    r2f = scores['rouge-2']['f']
    rouge_data.append([r1f, r2f])
    # bs, br = bestSummary(text, summary, c=3);
    ref_summary = summary
    sent_list = text;
    rouge = Rouge()
    ind_rouge = [];
    for sent in sent_list:
        scores = rouge.get_scores(sent, ref_summary)[0];
        ind_rouge.append(scores['rouge-1']['f'])
    ind_rouge = np.array(ind_rouge)


    # cmax = 10;
    # best_summary = '';
    # best_rouge = 0;
    # best_inds = [];
    # lower_bound = 3;
    # if (len(sent_list) < lower_bound):
    #     lower_bound = len(sent_list);
    # if(cmax > len(sent_list)):
    #     cmax = len(sent_list);
    # print(ind_rouge)
    # for summary_length in range(lower_bound, cmax + 1):
    #     ind = np.argpartition(ind_rouge, -1 * summary_length)[-1 * summary_length:]
    #     # print(ind)
    #     summary_prop = ' '.join([sent_list[i] for i in ind]);
    #     scores = rouge.get_scores(summary_prop, ref_summary)[0];
    #     rf1 = scores['rouge-1']['f'];
    #     if (rf1 > best_rouge):
    #         best_rouge = rf1;
    #         best_summary = summary_prop
    #         best_inds = ind;
    #
    # bs = best_summary;


    best_summary = ' '.join(sent_list[0:3]);

    best_rouge = rouge.get_scores(best_summary, ref_summary)[0]['rouge-1']['f'];
    best_ind = [0,1,2];
    counter = 0;

    for c in range(3,4):
        sent_inds = list(range(len(sent_list)))
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
            if(counter>200):
                break;
    bs = best_summary;
    best_inds = best_ind;




    br = rouge.get_scores(bs, summary)[0]['rouge-1']['f'];
    enhanced_rouge.append(br)
    ind_distribution += list(best_inds);
    extractive_summaries.append(bs);
    preserved_texts.append(text);
    preserved_summaries.append(summary)
    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
enhanced_rouge = np.array(enhanced_rouge)
rouge_data = np.array(rouge_data);
print(np.mean(rouge_data[:, 0]))  # baseline 0.305
print(np.mean(rouge_data[:, 1]))  # basline 0.203
print(np.mean(enhanced_rouge))

print(np.mean(sentence_lengths))  # man sentence length is 32

pickle.dump((preserved_summaries, preserved_texts, extractive_summaries),
            open('algorithmic_extraction_summaries_%d.p' % len(extractive_summaries), 'wb'))
plt.hist(sentence_lengths, bins=20)
plt.show()

plt.hist(enhanced_rouge, bins=20);
plt.show()

##distribution of selections
ind_distribution = dict(Counter(ind_distribution))
print(ind_distribution)
plt.hist(ind_distribution.values())
plt.show()