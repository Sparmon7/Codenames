import msgpack
import lzma
import numpy as np
import pandas as pd
import random

import time


# adjustable parameters
skill_level=30
assassin_penalty=-5


board_embeddings_cache = {}

def decode_numpy(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


with lzma.open("embeddings.msgpack.xz", "rb") as f:
    data = msgpack.unpackb(f.read(), object_hook=decode_numpy)


# Codenames words from https://boardgamegeek.com/thread/1413932/word-list
words = open('words.txt').read().replace('\n', ' ').split()
# convert to lowercase for compatibility with GloVe
words = [word.lower() for word in words]

# Create board, based on rules: https://czechgames.com/files/rules/codenames-rules-en.pdf
board_words = random.sample(words, 25)  # pick 25 words for the board, randomly
# red_words = board_words[:9] # 9 blue words (assuming blue is the starting team)
# blue_words = board_words[9:17] # 8 red words
# assassin_word = board_words[17] # 1 assassin word
# bystander_words = board_words[18:25] # 7 bystander words


def cosine_similarity(a, b): return np.dot(
    a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def check_validity(word, board_words):
    for i in board_words:
        if word in i or i in word:
            return False
    return True

def get_board_embeddings(): 
    embeddings = []
    for word in board_words:
        if word not in board_embeddings_cache:
            board_embeddings_cache[word] = data[word]
        embeddings.append(board_embeddings_cache[word])
    return np.vstack(embeddings)


def calculate_expected(word, softmaxes, guesses):
    temp = softmaxes.copy()
    if word < 9:
        if guesses == 1:
            return temp[word]
        else:
            expected_words = temp[word]
            prob = temp[word]
            temp[word] = 0
            temp = temp/np.sum(temp)
            for w in range(18):
                if temp[w] > 0:
                    expected_words += prob * \
                        calculate_expected(w, temp, guesses-1)
            return expected_words
    elif word >= 9 and word < 17:
        return -1*temp[word]
    elif word == 17:
        return assassin_penalty*temp[word]
    else:
        return 0


def calculate_expected_dp(idx, softmaxes, guesses):
    # memoization table
    dp = np.zeros((guesses + 1, 18))  # [remaining_guesses][word_idx]

    # base case for 1 guess
    for i in range(18):
        if i < 9:  # team A words
            dp[1][i] = softmaxes[i]
        elif i < 17:  # opponent words
            dp[1][i] = -softmaxes[i]
        elif i == 17:  # assassin
            dp[1][i] = assassin_penalty * softmaxes[i]
        # else: bystander words remain 0

    # fill dp table for multiple guesses
    for g in range(2, guesses + 1):
        for i in range(18):
            if i < 9:  # team A words for multiple guesses
                prob = softmaxes[i]

                next_probs = softmaxes.copy()
                next_probs[i] = 0
                total = next_probs.sum()
                if total > 0:
                    next_probs /= total
                # expected value
                dp[g][i] = prob * (
                    1 + sum(next_probs[j] * dp[g - 1][j] for j in range(18))
                )

    return dp[guesses][idx]


def get_best_clue_optimized(board_words, team_count, batch_size=200, verbose=False):
    best_max = -1
    best_clue = None
    best_guesses = None

    total = team_count + 9  # 8 opponent + 1 assassin added

    # recompute board vectors and their norms only once
    board_vecs = np.array([data[word] for word in board_words])
    board_norms = np.linalg.norm(board_vecs, axis=1)
    
    # cet all candidate clues that are valid
    candidates = [word for word in data.keys() if check_validity(word, board_words)]
    if verbose: 
        print(f"Found {len(candidates)} valid candidate clues")
    
    # Process candidates in batches
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        
        # get all candidate vectors at once
        batch_vecs = np.array([data[candidate] for candidate in batch])
        batch_norms = np.linalg.norm(batch_vecs, axis=1)
        
        similarities = np.dot(batch_vecs, board_vecs.T)

        batch_norms_reshaped = batch_norms.reshape(-1, 1)
        cos_sims = similarities / (batch_norms_reshaped * board_norms)

        for idx, candidate in enumerate(batch):
            cos_sim = cos_sims[idx]
            
            exp_scores = np.exp(skill_level * cos_sim)
            softmaxes = exp_scores / np.sum(exp_scores)
            
            # early filtering: only evaluate promising candidates
            # check if potential is too low based on team word similarities
            team_word_similarities = cos_sim[:team_count]
            if np.max(team_word_similarities) < 0.3 and len(candidates) > 1000:
                continue
                
            # Evaluate for different guesses
            for guesses in range(1, 4):
                expected_words = 0
                for word in range(total):
                    expected_words += calculate_expected_dp(word, softmaxes, guesses)
                    
                if expected_words > best_max:
                    best_max = expected_words
                    best_clue = candidate
                    best_guesses = guesses
                    
                    # Logging for progress tracking
                    if i % (batch_size * 10) == 0:
                        top_words = [board_words[k] for k in np.argsort(softmaxes)[::-1][:guesses]]
                        if verbose: 
                            print(f"New best: '{candidate}', expected: {expected_words:.3f}, guesses: {guesses}, words: {top_words}")
                    
    return best_clue, best_guesses

def simulate_game_optimized(verbose=True):
    """
    Simulate a game of Codenames with optimized performance.
    """
    start_time = time.time()
    
    team_total = 9
    team_words = random.sample(words, team_total)
    
    remaining = [w for w in words if w not in team_words]
    opponent_words = random.sample(remaining, 8)
    remaining = [w for w in remaining if w not in opponent_words]
    
    assassin_word = random.choice(remaining)
    remaining.remove(assassin_word)
    
    bystander_words = random.sample(remaining, 7)
    
    if verbose:
        print("\n==== NEW GAME ====")
        print(f"Team words: {team_words}")
        print(f"Opponent words: {opponent_words}")
        print(f"Assassin: {assassin_word}")
    
    # board ordering: first team_total words, then opponents, assassin, bystanders.
    board_words = team_words + opponent_words + [assassin_word] + bystander_words
    
    turns = 0
    current_team = team_words[:]  # list of words yet to be guessed
    
    while current_team:
        turns += 1
        turn_start = time.time()
        
        if verbose:
            print(f"\n-- Turn {turns} --")
            print(f"Remaining team words: {current_team}")
        
        board_words = current_team + opponent_words + [assassin_word] + bystander_words
        team_count = len(current_team)
        
        if verbose:
            print(f"Finding best clue for {team_count} team words...")
        
        # Use optimized clue selection
        clue, guesses = get_best_clue_optimized(board_words, team_count, verbose=False)
        
        actual = min(guesses, team_count)
        chosen_words = current_team[:actual]
        
        if verbose:
            print(f"Best clue: '{clue}', suggesting {guesses} guesses")
            print(f"Words guessed correctly: {chosen_words}")
            print(f"Turn completed in {time.time() - turn_start:.2f} seconds")
        
        current_team = current_team[actual:]
        
        if verbose and not current_team:
            print("\n✓ All team words found!")
    
    game_time = time.time() - start_time
    if verbose:
        print(f"\nGame completed in {turns} turns ({game_time:.2f} seconds)")
    
    return turns

def simulate_games_optimized(n_games, verbose=True):
    """
    Simulate multiple games with optimized performance.
    """
    start_time = time.time()
    results = []
    
    if verbose:
        print(f"Simulating {n_games} games with skill_level={skill_level}, assassin_penalty={assassin_penalty}")
    
    for i in range(n_games):
        if verbose:
            print(f"\nSimulating game {i+1}/{n_games}")
        
        # Only be verbose for the first game if simulating multiple
        game_verbose = verbose if n_games == 1 or i == 0 else False
        turns = simulate_game_optimized(verbose=game_verbose)
        results.append(turns)
        
        if verbose and n_games > 1:
            print(f"Completed {i+1}/{n_games} games. Avg turns so far: {sum(results)/(i+1):.2f}")
    
    # Compute distribution
    turns_distribution = {}
    for t in results:
        turns_distribution[t] = turns_distribution.get(t, 0) + 1
    
    print("\n=== SIMULATION RESULTS ===")
    print(f"Simulated {n_games} games with skill_level={skill_level}, assassin_penalty={assassin_penalty}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print("Distribution of turns to finish the game:")
    for t in sorted(turns_distribution.keys()):
        pct = (turns_distribution[t] / n_games) * 100
        print(f"Turns = {t}: {turns_distribution[t]} games ({pct:.1f}%)")
    
    avg_turns = sum(results) / len(results)
    print(f"\nAverage turns per game: {avg_turns:.2f}")
    
    return results

result = simulate_games_optimized(1, True)
