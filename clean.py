

import os
for file in os.listdir('./'):
    if('newsroom_train' in file):
        print(file)
        os.remove(file)