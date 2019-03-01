from newsroom import jsonl
dir = 'D:\\Documents\\Classes\\CS224n\\project'
import os
# Read entire file:

with jsonl.open(os.path.join(dir, 'thin\\train.data'), gzip = True) as train_file:
    train = train_file.read()

# Read file entry by entry:

with jsonl.open("./thin/train.data", gzip = True) as train_file:
    for entry in train_file:
        print(entry["summary"], entry["text"])