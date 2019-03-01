from newsroom import jsonl
import pickle;
# Read entire file:

with jsonl.open("train.data", gzip = True) as train_file:
    train = train_file.read()

# Read file entry by entry:

#mine this into training data/label = summary


training_articles = [];
training_labels = [];

with jsonl.open("train.data", gzip = True) as train_file:
    c = 1;
    for entry in train_file:
        if(c%10000 == 0):
            print("%d files processed"%c)
        #print(entry.keys()) #keys are url, archive, title, date, text, summary, density, coverge, comppression, compression bin, coverage, density"
        #print(len(entry["summary"]), len(entry["text"]))
        training_point = entry["summary"];
        training_tgt = entry["text"];
        training_articles.append(training_point);
        training_labels.append(training_tgt)

        if(c% 100000 ==0):
            pickle.dump((training_articles, training_labels), open("./processed_train/newsroom_train_%d.p"%c, "wb"))
            training_articles = [];
            training_labels = [];

        c+=1;

        