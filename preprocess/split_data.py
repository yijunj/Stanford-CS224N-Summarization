import pickle
import sys

# sys.path.append('/Users/yiliu/Desktop/CS224N/final_project/scripts')

with open('../dataset/extraction_summaries_3.p', 'rb') as pickle_file:
    data = pickle.load(pickle_file)

print(len(data))
print(len(data[0]))

total_len = len(data[0]) #73273 ; 72680

dev_len = 2000 #2000 ; 2000
test_len = 1680 #1273 ; 1680
train_len = total_len - dev_len - test_len

train_data = []
for i in range(4):
    train_data.append(data[i][:train_len])
with open('../dataset/extraction_summaries_3_train.p', 'wb') as pickle_file:
    pickle.dump(train_data, pickle_file)

dev_data = []
for i in range(4):
    dev_data.append(data[i][train_len:(train_len+dev_len)])
with open('../dataset/extraction_summaries_3_dev.p', 'wb') as pickle_file:
    pickle.dump(dev_data, pickle_file)

test_data = []
for i in range(4):
    test_data.append(data[i][(train_len+dev_len):])
with open('../dataset/extraction_summaries_3_test.p', 'wb') as pickle_file:
    pickle.dump(test_data, pickle_file)


with open('../dataset/extraction_summaries_3_train.p', 'rb') as pickle_file:
    train_data = pickle.load(pickle_file)

with open('../dataset/extraction_summaries_3_dev.p', 'rb') as pickle_file:
    dev_data = pickle.load(pickle_file)

with open('../dataset/extraction_summaries_3_test.p', 'rb') as pickle_file:
    test_data = pickle.load(pickle_file)

print(len(train_data))
print(len(train_data[0]))
print('============')
print(len(dev_data))
print(len(dev_data[0]))
print('============')
print(len(test_data))
print(len(test_data[0]))
