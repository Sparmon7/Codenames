import msgpack
import lzma
import numpy as np
import pandas as pd
import random
import time
import math
import sys

#load words
def decode_numpy(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj
print('Welcome to the CodeNames Assistant, the bot that helps you give clues to win the game CodeNames')
print('Loading dictionary of words')
with lzma.open("embeddings.msgpack.xz", "rb") as f:
    data = msgpack.unpackb(f.read(), object_hook=decode_numpy)

#get board layout
def check_real_word(word):
    return word in data

#take in words from user
def begin():    
    num_input = -1
    while num_input != 8 and num_input !=9:
        print("How many words does your team have?")
        num_input = int(input())
    good_words=[]
    bad_words=[]
    assassin_word=[]
    bystander_words=[]

    print("Type in your team's words, pressing enter after each word")
    for i in range(num_input):
        word = input().lower()
        valid = check_real_word(word)
        while not valid:
            print('That word is invalid, please try again')
            word = input().lower()
            valid = check_real_word(word)
        good_words.append(word)
        
    print("Type in the other team's words, pressing enter after each word")
    for i in range(17-num_input):
        word = input().lower()
        valid = check_real_word(word)
        while not valid:
            print('That word is invalid, please try again')
            word = input().lower()
            valid = check_real_word(word)
        bad_words.append(word)
        
    print("Type in the assassin word")
    word = input().lower()
    valid = check_real_word(word)
    while not valid:
        print('That word is invalid, please try again')
        word = input().lower()
        valid = check_real_word(word)
    assassin_words=[word]

    print("Type in the bystander words, pressing enter after each word")
    for i in range(7):
        word = input().lower()
        valid = check_real_word(word)
        while not valid:
            print('That word is invalid, please try again')
            word = input().lower()
            valid = check_real_word(word)
        bystander_words.append(word)

    return good_words, bad_words, assassin_words, bystander_words

def cosine_similarity(a, b): return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

#make sure clue isn't in one of the words
def check_validity(word, board_words):
    for i in board_words:
        if word in i or i in word:
            return False
    return True

#make sure possible clues are associated with a word
def check_minimum_threshold(word, good_words, threshold = 0.45):
    for i in good_words:
        if cosine_similarity(word,data[i])>threshold:
            return True        
    return False

#create preliminary list of potential clues for use through the rest of the game
def generate_inital_clues(good_words, bad_words, assassin_words, bystander_words, skill_level=15):
    print('Generating initial clues')
    clues = {}
    board_words = [*good_words, *bad_words, *assassin_words, *bystander_words]
    for i in data:
        if check_validity(i, board_words) and check_minimum_threshold(data[i], good_words):
            softmaxes = [math.exp(skill_level*cosine_similarity(data[word], data[i])) for word in board_words]
            softmaxes/=np.sum(softmaxes)
            clues[i]=softmaxes
    return clues

#gives guess based on softmaxes
def generate_guess(clues, good_words, bad_words, assassin_words, bystander_words):
    best_clue = None
    best_guesses=-1
    max = -10
    for word in clues:
        softmaxes = clues[word]
        for guesses in range(1,6):
            expected_words = 0
            for poss in range(18):
                expected_words+=calculate_expected(poss, softmaxes, guesses, len(good_words))
            if expected_words>max: 
                max=expected_words
                best_clue=word
                best_guesses = guesses
    
    # #printing expected words           
    # words=[*good_words, *bad_words, *assassin_words, *bystander_words]
    # print([np.array(clues[best_clue]).argsort()[-1*best_guesses:][::-1]])
    # print(clues[best_clue][[np.array(clues[best_clue]).argsort()[-1*best_guesses:][::-1]]])
    # print([words[i] for i in np.array(clues[best_clue]).argsort()[-1*best_guesses:][::-1]])
    
    del clues[best_clue]
    return best_clue,best_guesses,clues

#expected number of words for a guess
def calculate_expected(word, softmaxes, guesses, good_length, assassin_penalty=-9):
    temp = softmaxes.copy()
    if word <good_length:
        if guesses == 1:
            return temp[word]
        else:
            expected_words = temp[word]
            prob = temp[word]
            temp[word]=0
            temp=temp/np.sum(temp)
            for w in range(18):
                if temp[w]>0:
                    expected_words+=prob*calculate_expected(w, temp, guesses-1, good_length)
            return expected_words
    elif word >=good_length and word<17:
        return -1*temp[word]
    elif word == 17:
        return assassin_penalty*temp[word]
    else:
        return 0

#remove guessed words
def remove_words(clues, good_words, bad_words, assassin_words, bystander_words, turn):
    board_words = [*good_words, *bad_words, *assassin_words, *bystander_words]
    product = np.ones(25)
    if turn:
        print('Type the words that are guessed for your turn, pressing enter after each and enter after the turn is done')
    else:
        print('Type the words that are guessed for the opponents\' turn and enter after the turn is done')
    while True:
        word = input().lower()
        if word in board_words:
            product[board_words.index(word)]=0
        elif word == "":
            break
        else:
            print("This word was not found, please try again")
    removals = []
    for i in clues:
        clues[i]= clues[i]*product
        clues[i]/=np.sum(clues[i])
        remove=True
        #remove less than 0.45
        for j in range(len(good_words)):
            if cosine_similarity(data[good_words[j]], data[i])>0.45 and clues[i][j]>0:
                remove=False
        if remove:
            removals.append(i)
    for i in removals:
        del clues[i]
    return clues
            
                

# #for preloading random words to save time for testing
# good = ["mammoth", "racket", "school", "worm", "nut", "microscope", "fork", "chest", "mole"]
# bad = ["press", "plot", "tail", "soldier", "gas", "button", "agent", "flute"]
# assassin = ["time"]
# bystander = ["america", "buffalo", "field", "tube", "ghost", "grass", "dwarf"]

good, bad, assassin, bystander= begin()
clues = generate_inital_clues(good,bad,assassin,bystander)
team_turn = len(good)==9
done = False
while not done:
    if team_turn:
        best_clue, best_guesses, clues = generate_guess(clues, good, bad, assassin, bystander)
        print(f"Suggested guess: {best_clue} for {best_guesses}")
        clues = remove_words(clues, good, bad, assassin, bystander, True)   
    else:
        clues = remove_words(clues, good, bad, assassin, bystander, False)
    team_turn = not team_turn