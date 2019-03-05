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
ind_distribution= [];
extractive_summaries = [];
preserved_texts = []; preserved_summaries = [];
for c in range(len(summaries)):
    # if(c > 200):
    #     break;
    if(c%10 == 1):
        print('processed %d samples' % c)
        print(np.mean([len(s) for s in preserved_summaries]))
        print(np.mean(enhanced_rouge))
    if(c == 2000):
        break;
    # if((c+1)%25000 == 0):
    #     pickle.dump((preserved_summaries, preserved_texts, extractive_summaries),
    #                 open('algorithmic_extraction_summaries_%d.p' % len(extractive_summaries), 'wb'))

    text = texts[c]; summary = summaries[c];
    if(len(text) <= 1):
        continue;
    first3Sentences = ' '.join(text[0:num_sents_summary])
    #print(first3Sentences)
    sentence_lengths.append(len(text))
    try:
        scores = rouge.get_scores(first3Sentences, summary)[0];
        # print(scores)
        r1f = scores['rouge-1']['f']
        r2f = scores['rouge-2']['f']
        rouge_data.append([r1f, r2f])
        #bs, br = bestSummary(text, summary, c=3);
        #bs, br, ind = bestSummary_individual(text, summary, cmax=10)
        #bs, br, ind = bestSummary(text, summary)
        bs, br, ind = bestSummary_iterative(text, summary)


        #br = rouge.get_scores(bs, summary)[0]['rouge-1']['f'];
        enhanced_rouge.append(br)
        ind_distribution += list(ind);
        extractive_summaries.append(bs);
        preserved_texts.append(text);
        preserved_summaries.append(summary)
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
            open('algorithmic_extraction_summaries_%d.p' % len(extractive_summaries), 'wb'))
pickle.dump((sentence_lengths, enhanced_rouge, ind_distribution), open('extraction_data.p', 'wb'))

plt.hist(sentence_lengths, bins = 20)
plt.show()

plt.hist(enhanced_rouge, bins = 20);
plt.show()

##distribution of selections
ind_distribution= dict(Counter(ind_distribution))
print(ind_distribution)
plt.hist(ind_distribution.values())
plt.show()