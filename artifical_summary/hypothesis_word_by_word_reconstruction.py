import sys
from rouge import Rouge
import pickle
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
    if(c%1000 == 0):
        print('processed %d samples' % c)
    if((c+1)%25000 == 0):
        pickle.dump((preserved_summaries, preserved_texts, extractive_summaries),
                    open('algorithmic_extraction_summaries_%d.p' % len(extractive_summaries), 'wb'))

    text = texts[c]; summary = summaries[c];
