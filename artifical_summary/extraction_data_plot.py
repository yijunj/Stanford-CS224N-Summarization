import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter;
import seaborn as sns
from helper_functions.core_plot import *
# (sentence_lengths, enhanced_rouge, ind_distribution)
(sentence_lengths, enhanced_rouge, ind_distribution)= pickle.load(open('extraction_data_9c(2,3,4).p', 'rb'))
print(ind_distribution)
(sentence_lengthsf, enhanced_rougef, ind_distributionf)= pickle.load(open('extraction_data_fast.p', 'rb'))
(sentence_lengthsi, enhanced_rougei, ind_distributioni)= pickle.load(open('extraction_data_iterative_50.p', 'rb'))
ind_dist_values =[i for i in ind_distribution];
ind_distributionf= dict(Counter(ind_distributionf))
print(ind_distributionf)
#ind_dist_valuesf = list(ind_distributionf.values());
ind_dist_valuesi =ind_distributioni
#ind_dist_valuesf = list(ind_distributionf.values());
ind_dist_valuesf =[i for i in ind_distributionf];

sns.set(color_codes=True)

plt.figure(figsize = (8.5,5.5));
ax1 =plt.subplot(231)
#plt.hist(enhanced_rouge, bins = 20);
sns.distplot(enhanced_rouge, bins = 20, kde = False, hist_kws=dict(alpha=0.5));
plt.xlabel('Rouge-1 f')
plt.title('exhaustive')
plt.ylabel('counts')

ax2 = plt.subplot(232)


sns.distplot(enhanced_rougef, bins = 20, kde = False, hist_kws=dict(alpha=0.5), color = 'green');
plt.xlabel('Rouge-1 f')
plt.title('k-best')

ax3 = plt.subplot(233)
#plt.hist(enhanced_rouge, bins = 20);
plt.title('iterative k-best')
sns.distplot(enhanced_rougei, bins = 20, kde = False, hist_kws=dict(alpha=0.5), color = 'orange');

plt.xlabel('Rouge-1 f')


ax4 = plt.subplot(234)
sns.distplot(ind_dist_values, bins = 8, kde = False, hist_kws=dict(alpha=0.5));
plt.xlabel('sentence index')
plt.ylabel('counts')
##distribution of selections

ax5 = plt.subplot(235)
sns.distplot(ind_dist_valuesf, bins = 20, kde = False, hist_kws=dict(alpha=0.5), color = 'green');
plt.xlabel('sentence index')


ax6 = plt.subplot(236)

sns.distplot(ind_dist_valuesi, bins = 20, kde = False, hist_kws=dict(alpha=0.5), color = 'orange');
plt.xlabel('sentence index')
apply_sublabels([ax1, ax2, ax3, ax4, ax5, ax6])
plt.tight_layout();
plt.savefig('extraction_data.png')
plt.show()

print(np.mean(enhanced_rougei))