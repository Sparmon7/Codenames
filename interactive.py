import msgpack
import lzma
import numpy as np
import random
import time
import math
import sys
from monte_carlo_generator import MonteCarloClueGenerator
from utils import (
    load_embeddings,
    cosine_similarity,
    check_real_word,
    check_minimum_threshold,
    check_validity,
    remove_words_monte_carlo
)

similarity_cache = {}

# take in words from user
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

# create preliminary list of potential clues for use through the rest of the game
def generate_inital_clues(good_words, bad_words, assassin_words, bystander_words, skill_level=15):
    start = time.time()
    
    print('Generating initial clues')
    clues = {}
    board_words = [*good_words, *bad_words, *assassin_words, *bystander_words]
    
    # prebuild word embeddings and norms for board and good words
    board_embeddings = np.stack([data[word] for word in board_words])
    board_norms = np.linalg.norm(board_embeddings, axis=1)
    
    good_embeddings = np.stack([data[word] for word in good_words])
    good_norms = np.linalg.norm(good_embeddings, axis=1)

    valid_candidates = [c for c in data if check_validity(c, board_words)]

    for candidate in valid_candidates:
        # candidates that contain any board word (or vice versa)
        if not all(candidate not in bw and bw not in candidate for bw in board_words):

            continue
        candidate_embedding = data[candidate]
        candidate_norm = np.linalg.norm(candidate_embedding)
        
        # cosine similarities to good words and then bad wrods
        cosines = (good_embeddings @ candidate_embedding) / (good_norms * candidate_norm)
        if not check_minimum_threshold(candidate, good_words):
            continue

        sims = (board_embeddings @ candidate_embedding) / (board_norms * candidate_norm)
        exps = np.exp(skill_level * sims)
        softmaxes = exps / np.sum(exps)
        
        clues[candidate] = softmaxes

    end = time.time()
    print("Clue generation took {:.4f} seconds".format(end - start))
    return clues

#gives guess based on softmaxes
def generate_guess(clues, good_words, bad_words, assassin_words, bystander_words):
    print("Generating best guess for current turn...")
    best_clue = None
    best_guesses = -1
    max_value = -10
    for word in clues:
        softmaxes = clues[word]
        for guesses in range(1, 6):
            expected_words = 0
            for poss in range(18):
                expected_words += calculate_expected(poss, softmaxes, guesses, len(good_words))
            if expected_words > max_value:
                max_value = expected_words
                best_clue = word
                best_guesses = guesses

    # Uncomment the below lines for debugging expected words details
    # words = [*good_words, *bad_words, *assassin_words, *bystander_words]
    # print([np.array(clues[best_clue]).argsort()[-1 * best_guesses:][::-1]])
    # print(clues[best_clue][[np.array(clues[best_clue]).argsort()[-1 * best_guesses:][::-1]]])
    # print([words[i] for i in np.array(clues[best_clue]).argsort()[-1 * best_guesses:][::-1]])

    del clues[best_clue]
    return best_clue, best_guesses, clues

#expected number of words for a guess
def calculate_expected(word, softmaxes, guesses, good_length, assassin_penalty=-9):
    temp = softmaxes.copy()
    if word < good_length:
        if guesses == 1:
            return temp[word]
        else:
            expected_words = temp[word]
            prob = temp[word]
            new_temp = np.copy(temp)
            new_temp[word] = 0
            total = np.sum(new_temp)
            if total > 0:
                new_temp /= total
                valid = np.nonzero(new_temp)[0]
                expected_words += prob * sum(calculate_expected(w, new_temp, guesses-1, good_length) for w in valid)
            return expected_words
    elif word >= good_length and word < 17:
        return -1 * temp[word]
    elif word == 17:
        return assassin_penalty * temp[word]
    else:
        return 0

#remove guessed words
def remove_words(clues, good_words, bad_words, assassin_words, bystander_words, turn):
    board_words = [*good_words, *bad_words, *assassin_words, *bystander_words]
    product = np.ones(25)
    if turn:
        print('Type the words that are guessed for your turn, pressing enter after each word and enter after the turn is done')
    else:
        print('Type the words that are guessed for the opponents\' turn and enter after the turn is done')
    while True:
        word = input().lower()
        if word == "":
            break
        if word in board_words:
            # remove
            if word in good_words:
                good_words.remove(word)
            if word in bad_words:
                bad_words.remove(word)
            if word in assassin_words:
                assassin_words.remove(word)
            if word in bystander_words:
                bystander_words.remove(word)
                
            # Set corresponding product index to 0, based on original board_words order.
            idx = board_words.index(word)
            product[idx] = 0
            
            # update board_words so duplicate words aren't removed twice.
            board_words[idx] = None
        else:
            print("This word was not found, please try again")
    
    removals = []
    for candidate in list(clues.keys()):
        clues[candidate] = clues[candidate] * product
        if np.sum(clues[candidate]) > 0:
            clues[candidate] /= np.sum(clues[candidate])
        else:
            clues[candidate] = np.zeros_like(clues[candidate])
        remove = True
        for i in range(len(good_words)):
            if cosine_similarity(data[good_words[i]], data[candidate]) > 0.45 and clues[candidate][i] > 0:
                remove = False
                break
        if remove:
            removals.append(candidate)

    for candidate in removals:
        del clues[candidate]
    return clues, good_words, bad_words, assassin_words, bystander_words

def generate_board_random(team_count=9):
    if team_count not in (8, 9):
        raise ValueError("team_count must be either 8 or 9")
    
    with open("words.txt", "r") as file:
        words = [line.strip().replace(" ", "").lower() for line in file if line.strip()]
    total_words = 25
    opponent_count = 17 - team_count  # 8 or 9, depending on team_count
    assassin_count = 1
    bystander_count = total_words - team_count - opponent_count - assassin_count  # always 7
    
    board = random.sample(words, total_words)
    
    good_words = board[:team_count]
    bad_words = board[team_count:team_count+opponent_count]
    assassin_words = [board[team_count+opponent_count]]
    bystander_words = board[team_count+opponent_count+assassin_count:]
    
    return good_words, bad_words, assassin_words, bystander_words

def begin_automate():
    print("Would you like to enter the board manually or generate it randomly?")
    print("Type 'm' for manual entry or 'r' to generate a board:")
    choice = input().lower().strip()
    
    if choice == "r":
        num_input = -1
        while num_input not in (8, 9):
            print("How many words does your team have? (8 or 9)")
            try:
                num_input = int(input())
            except ValueError:
                continue
        good_words, bad_words, assassin_words, bystander_words = generate_board_random(num_input)
        
        print("\nGenerated Board:")
        print("Your team words:", good_words)
        print("Other team words:", bad_words)
        print("Assassin word:", assassin_words)
        print("Bystander words:", bystander_words)
        print("\n")
        
        return good_words, bad_words, assassin_words, bystander_words
    
    return begin()
                
# #for preloading random words to save time for testing
# good = ["mammoth", "racket", "school", "worm", "nut", "microscope", "fork", "chest", "mole"]
# bad = ["press", "plot", "tail", "soldier", "gas", "button", "agent", "flute"]
# assassin = ["time"]
# bystander = ["america", "buffalo", "field", "tube", "ghost", "grass", "dwarf"]

def main(): 
    print("\nStarting new Codenames game...")

    print("Run with monte carlo? (type y or n)")
    monte_carlo = input().strip().lower() == 'y'

    good, bad, assassin, bystander = begin_automate()
    if monte_carlo:
        mc_generator = MonteCarloClueGenerator(n_simulations=100)    
    else: 
        global data
        data = load_embeddings()

        clues = generate_inital_clues(good,bad,assassin,bystander)

    team_turn = len(good)==9
    done = False

    turn_count = 0
    while not done:
        if team_turn:
            print("\nGenerating best clue...")
            print("\nYour turn!")
           
            start_time = time.time()
           
            if monte_carlo:
                best_clue, best_guesses = mc_generator.generate_best_clue(good, bad, assassin, bystander)
            else: 
                best_clue, best_guesses, clues = generate_guess(clues, good, bad, assassin, bystander)

            elapsed_time = time.time() - start_time

            print(f"\nTime to generate clue: {elapsed_time:.2f} seconds")
            print(f"Suggested guess: {best_clue} for {best_guesses} guesses")

            if monte_carlo: 
                good, bad, assassin, bystander = remove_words_monte_carlo(good, bad, assassin, bystander, True)
            else: 
                clues, good, bad, assassin, bystander = remove_words(clues, good, bad, assassin, bystander, True)
        else:
            print("\nOpponent's turn!")
            if monte_carlo:
                good, bad, assassin, bystander = remove_words_monte_carlo(good, bad, assassin, bystander, False)
            else: 
                clues, good, bad, assassin, bystander = remove_words(clues, good, bad, assassin, bystander, True)

        team_turn = not team_turn
        turn_count += 1

        if turn_count % 2 == 0:
            print("\nCurrent Board State:")
            print("Remaining team words:", good)
            print("Remaining other team words:", bad)
            print("Remaining assassin word:", assassin)
            print("Remaining bystander words:", bystander)
            print('\n')
            
            # Check for game end conditions
            if not good:
                print("Game Over! Your team has won!")
                done = True
            elif not bad:
                print("Game Over! The other team has won!")
                done = True
            elif not assassin:
                print("Game Over! The assassin was found!")
                done = True

if __name__ == "__main__": 
    main()
