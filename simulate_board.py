import msgpack
import lzma
import numpy as np
import pandas as pd
import random
import math

#adjustable parameters
skill_level=25
assassin_penalty=-9


def decode_numpy(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj

with lzma.open("embeddings.msgpack.xz", "rb") as f:
    data = msgpack.unpackb(f.read(), object_hook=decode_numpy)
    


# Codenames words from https://boardgamegeek.com/thread/1413932/word-list
words = open('words.txt').read().replace('\n',' ').split()
words = [word.lower() for word in words] # convert to lowercase for compatibility with GloVe

# Create board, based on rules: https://czechgames.com/files/rules/codenames-rules-en.pdf
board_words = random.sample(words, 25) # pick 25 words for the board, randomly 
# red_words = board_words[:9] # 9 blue words (assuming blue is the starting team)
# blue_words = board_words[9:17] # 8 red words
# assassin_word = board_words[17] # 1 assassin word
# bystander_words = board_words[18:25] # 7 bystander words


cosine_similarity = lambda a,b: np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def check_validity(word, board_words):
    for i in board_words:
        if word in i or i in word:
            return False
    return True

def calculate_expected(word, softmaxes, guesses):
    temp = softmaxes.copy()
    if word <9:
        if guesses == 1:
            return temp[word]
        else:
            expected_words = temp[word]
            prob = temp[word]
            temp[word]=0
            temp=temp/np.sum(temp)
            for w in range(18):
                if temp[w]>0:
                    expected_words+=prob*calculate_expected(w, temp, guesses-1)
            return expected_words
    elif word >=9 and word<17:
        return -1*temp[word]
    elif word == 17:
        return assassin_penalty*temp[word]
    else:
        return 0


max = -1
best_clue = None
best_guesses = -1   
for i in data:
    if check_validity(i, board_words):
        softmaxes = [math.exp(skill_level*cosine_similarity(data[word], data[i])) for word in board_words]
        softmaxes = softmaxes/np.sum(softmaxes)
        for guesses in range(1,4):
            expected_words = 0
            for word in range(18):
                expected_words+=calculate_expected(word, softmaxes, guesses)
            if expected_words>max: 
                print(i, expected_words,guesses, [board_words[k] for k in np.argsort(softmaxes)[-1*guesses:][::-1]])
                max=expected_words
                best_clue=i
                best_guesses = 1