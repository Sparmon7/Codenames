import time
import numpy as np
import random
from utils import load_embeddings, cosine_similarity
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor


"""
General layout of things: 

1. inital setup: 
    initialize with number of simulations and load embeddings

2. For each candidate clue:
   - calculate similarities between clue and all board words

    - for each possible number of guesses (1-3):
     - Run multiple simulations (1000 good??):
       - sample guesses based on similarity probabilities
       - simulate game state after guesses and give score
     
     - calculate average score across simulations

3. Select best clue and number of guesses based on highest average score

stuff I added for performance sake:
    - make early wins have better scores 
    - stop early if clue is consistently worse than prev
    - parallelized simulations
"""

CLUE_LIMIT = 20
EARLY_STOP_THRESHOLD = 0.8  # stop if score is 80% of best so far, probably will have to adjust this later 

class MonteCarloClueGenerator:
    def __init__(self, n_simulations=1000, n_workers=4):
        print(f"\nInitializing Monte Carlo generator with {n_simulations} simulations...")
        self.n_simulations = n_simulations
        self.n_workers = n_workers
        self.data = load_embeddings()
        self._similarity_cache = {}
        
    @lru_cache(maxsize=1000) # TODO: test if this needs to be higher
    def get_word_similarities(self, clue, board_words_tuple):
        """Get similarity scores between clue and all board."""
        # added cache to save on calculations if the score is already computed
        board_words = list(board_words_tuple)
        return np.array([cosine_similarity(
            self.data[clue],
            self.data[word]
        ) for word in board_words])
    
    def sample_guesses(self, similarities, n_guesses):
        # Convert similarities to probabilities using softmax
        exp_sims = np.exp(similarities)
        probs = exp_sims / np.sum(exp_sims)
        
        # sample without replacement
        return np.random.choice(
            len(probs),
            size=min(n_guesses, len(probs)),
            replace=False,
            p=probs
        )
    
    def simulate_guesses(self, guesses, board_words, good_words, bad_words, assassin_words, bystander_words):
        # using sets for faster lookup, since there are going to be a lot of sims 
        good_set = set(good_words)
        bad_set = set(bad_words)
        assassin_set = set(assassin_words)
        bystander_set = set(bystander_words)
        
        for guess_idx in guesses:
            word = board_words[guess_idx]
            if word in good_set:
                good_set.remove(word)
            elif word in bad_set:
                bad_set.remove(word)
            elif word in assassin_set:
                assassin_set.remove(word)
            elif word in bystander_set:
                bystander_set.remove(word)
                
        return list(good_set), list(bad_set), list(assassin_set), list(bystander_set)
    
    def evaluate_board_state(self, good_words, bad_words, assassin_found=False, turn_count=0):
        """Score to evaluate board state from in the eyes of current player."""
        base_score = len(good_words) - len(bad_words)
        
        # penalize if assassin was found
        if assassin_found:
            return -float('inf') # might have to adjust this
            
        # reward early wins
        if not bad_words:
            return base_score + (10 - turn_count)  # more points for faster wins
            
        return base_score
    
    def simulate_game(self, clue, n_guesses, board_words, good_words, bad_words, assassin_words, bystander_words, max_turns=5):
        # converting to tuple for caching
        board_words_tuple = tuple(board_words)
        similarities = self.get_word_similarities(clue, board_words_tuple)
        
        # simulate our turn
        our_guesses = self.sample_guesses(similarities, n_guesses)
        current_good, current_bad, current_assassin, current_bystander = self.simulate_guesses(
            our_guesses, board_words, good_words, bad_words, assassin_words, bystander_words
        )
        
        assassin_found = not current_assassin  # empty list means assassin was found
        
        # simulate opponent if they have words left
        if current_bad and not assassin_found:  # only simulate opponent if game isn't over
            opponent_guesses = self.sample_guesses(similarities, random.randint(1, 2))
            current_good, current_bad, current_assassin, current_bystander = self.simulate_guesses(
                opponent_guesses, board_words, current_good, current_bad, current_assassin, current_bystander
            )
            
            assassin_found = assassin_found or not current_assassin
        
        # Calculate turn count based on remaining words
        turn_count = (len(good_words) - len(current_good)) + (len(bad_words) - len(current_bad))
        
        return self.evaluate_board_state(current_good, current_bad, assassin_found, turn_count)
    
    def simulate_game_parallel(self, args):
        """just a wrapper for the parallel executor"""
        return self.simulate_game(*args)

    def generate_best_clue(self, good_words, bad_words, assassin_words, bystander_words):
        print("\nGenerating best clue...")
        board_words = [*good_words, *bad_words, *assassin_words, *bystander_words]
        best_clue = None
        best_guesses = 0
        best_score = -float('inf')
        
        # board words tuple for caching
        board_words_tuple = tuple(board_words)
        
        # limit to top 20 clues (const defined on top of file)
        print("Finding initial candidate clues...")
        initial_scores = []
        for clue in self.data:
            if not any(clue in word or word in clue for word in board_words):
                similarities = self.get_word_similarities(clue, board_words_tuple)
                score = np.mean(similarities[:len(good_words)])
                initial_scores.append((clue, score))
        
        top_clues = sorted(initial_scores, key=lambda x: x[1], reverse=True)[:CLUE_LIMIT]
        print(f"Found {len(top_clues)} candidate clues")
        
        print("Simulating games for each candidate...")
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            for i, (clue, _) in enumerate(top_clues, 1):
                print(f"\nEvaluating clue {i}/{len(top_clues)}: {clue}")
                for n_guesses in range(1, 4):
                    # args for parallel execution
                    args = [(clue, n_guesses, board_words, good_words, bad_words, 
                            assassin_words, bystander_words) for _ in range(self.n_simulations)]
                    
                    # map simulated games across workers
                    scores = list(executor.map(self.simulate_game_parallel, args))
                    avg_score = np.mean(scores)
                    print(f"  Guesses: {n_guesses}, Average score: {avg_score:.2f}")
                    
                    # stpo early if score is worse
                    if avg_score < best_score * EARLY_STOP_THRESHOLD:
                        print(f"  Early stopping - score too low")
                        break
                        
                    if avg_score > best_score:
                        best_score = avg_score
                        best_clue = clue
                        best_guesses = n_guesses
        
        print(f"\nBest clue found: {best_clue} with {best_guesses} guesses (score: {best_score:.2f})")
        return best_clue, best_guesses 