import sys
from rouge import Rouge
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from artifical_summary.sentence_combination import *
from collections import Counter
sys.path.append("D:\Documents\Classes\CS224n\project")
dir = "D:\Documents\Classes\CS224n\project"
rouge = Rouge()

## cycle through articles

data = os.path.join(dir, "processed_train/newsroom_train_99844_word_lvl.p");
(summaries, texts) = pickle.load(open(data, 'rb'));
print(len(texts))
print(len(summaries))



rouge_data = [];
sentence_lengths = [];
num_sents_summary = 3;
enhanced_rouge = [];
ind_distribution= [];
extractive_summaries = [];
preserved_texts = []; preserved_summaries = [];
extractive_golden_inds = []
max_probe_length = 20
for c in range(len(summaries)):

    if(c%100 == 1):
        print('processed %d samples' % c)
        print(np.mean([len(s) for s in preserved_summaries]))
        print(np.mean(enhanced_rouge))
    # if(c == 2000):
    #     break;
    if((c+1)%500 == 0):
        pickle.dump((preserved_summaries, preserved_texts, extractive_summaries, extractive_golden_inds),
                    open('extraction_summaries_%d.p' % len(extractive_summaries), 'wb'))

    text = texts[c]; summary = summaries[c];
    if(len(text) <= 1):
        continue;

    ## =====================USE THESE FOR SENTENCE LEVEL PROCESSING =====================================##
    summary = ' '.join(summary);
    ## summaries come as list[list[list[str]]]
    first3Sentences = text[0:num_sents_summary];
    #list of strings
    text_as_list_of_strings = [' '.join(i) for i in text]
    first3Sentences = '. '.join([' '.join(i) for i in first3Sentences]);

    #first3Sentences = ' '.join(text[0:num_sents_summary])
    #print(first3Sentences)
    sentence_lengths.append(len(text))
    try:
        scores = rouge.get_scores(first3Sentences, summary)[0];
        # print(scores)
        r1f = scores['rouge-1']['f']
        r2f = scores['rouge-2']['f']
        rouge_data.append([r1f, r2f])

        bs, br, golden_inds = bestSummary_iterative(text_as_list_of_strings, summary, max_probe_length = max_probe_length)
        if(not(golden_inds) or len(golden_inds) != num_sents_summary):
            continue;
        enhanced_rouge.append(br)
        ind_distribution += list(golden_inds);
        extractive_summaries.append(bs);
        preserved_texts.append(text[0:max_probe_length]);
        preserved_summaries.append(summary)
        extractive_golden_inds.append(golden_inds)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
enhanced_rouge = np.array(enhanced_rouge)
rouge_data = np.array(rouge_data);
print(np.mean(rouge_data[:,0])) #baseline 0.305
print(np.mean(rouge_data[:,1])) #basline 0.203
print(np.mean(enhanced_rouge))

print(np.mean(sentence_lengths)) #man sentence length is 32

pickle.dump((preserved_summaries, preserved_texts, extractive_summaries),
            open('extraction_summaries_%d.p' % len(extractive_summaries), 'wb'))
pickle.dump((sentence_lengths, enhanced_rouge, ind_distribution), open('extraction_data.p', 'wb'))

