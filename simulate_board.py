import msgpack
import lzma
import numpy as np
import pandas as pd
import random

def decode_numpy(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj

with lzma.open("embeddings.msgpack.xz", "rb") as f:
    data = msgpack.unpackb(f.read(), object_hook=decode_numpy)

cosine_similarity = lambda a,b: np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# Codenames words from https://boardgamegeek.com/thread/1413932/word-list
words = open('words.txt').read().replace('\n',' ').split()
words = [word.lower() for word in words] # convert to lowercase for compatibility with GloVe

# Create board, based on rules: https://czechgames.com/files/rules/codenames-rules-en.pdf
board_words = random.sample(words, 25) # pick 25 words for the board, randomly 
red_words = board_words[:9] # 9 blue words (assuming blue is the starting team)
blue_words = board_words[9:17] # 8 red words
assassin_word = board_words[17] # 1 assassin word
bystander_words = board_words[18:25] # 7 bystander words

# Iterate through all pairs of red words. Find word that leads to good score for 2 words
best_score = 0
for i in data:
    for j in range(len(red_words)):
        for k in range(j + 1, len(red_words)):
            if not(red_words[j] in i or red_words[k] in i): # make sure word does not contain either of the 2 red words
                sim1 = cosine_similarity(data[red_words[j]],data[i])
                sim2 = cosine_similarity(data[red_words[k]],data[i])
                if sim1 > 0.45 and sim2 > 0.45:
                    print(i,red_words[j],red_words[k],sim1,sim2)

print("done")
