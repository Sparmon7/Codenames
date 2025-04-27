import time
import numpy as np
import random
from utils import load_embeddings, cosine_similarity
from functools import lru_cache

CLUE_LIMIT = 20

class MonteCarloClueGenerator:
    def __init__(self, n_simulations=1000):
        print(f"\nInitializing Monte Carlo generator with {n_simulations} simulations...")
        self.n_simulations = n_simulations
        self.data = load_embeddings()
        self._similarity_cache = {}
        
    @lru_cache(maxsize=1000)
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
        """Optimized board state simulation using sets for faster lookups."""
        # Convert to sets for O(1) lookups
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
    
    def evaluate_board_state(self, good_words, bad_words):
        """Score to evaluate board state from in the eyes of current player."""
        return len(good_words) - len(bad_words)
    
    def simulate_game(self, clue, n_guesses, board_words, good_words, bad_words, assassin_words, bystander_words, max_turns=5):
        # converting to tuple for caching
        board_words_tuple = tuple(board_words)
        similarities = self.get_word_similarities(clue, board_words_tuple)
        
        # simulate our turn
        our_guesses = self.sample_guesses(similarities, n_guesses)
        current_good, current_bad, current_assassin, current_bystander = self.simulate_guesses(
            our_guesses, board_words, good_words, bad_words, assassin_words, bystander_words
        )
        
        # simulate opponent if they have words left
        if current_bad:
            opponent_guesses = self.sample_guesses(similarities, random.randint(1, 2))
            current_good, current_bad, current_assassin, current_bystander = self.simulate_guesses(
                opponent_guesses, board_words, current_good, current_bad, current_assassin, current_bystander
            )
        
        return self.evaluate_board_state(current_good, current_bad)
    
    def generate_best_clue(self, good_words, bad_words, assassin_words, bystander_words):
        print("\nGenerating best clue...")
        board_words = [*good_words, *bad_words, *assassin_words, *bystander_words]
        best_clue = None
        best_guesses = 0
        best_score = -float('inf')
        
        # board words tuple for caching
        board_words_tuple = tuple(board_words)
        
        # Limit to top 20 clues
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
        for i, (clue, _) in enumerate(top_clues, 1):
            print(f"\nEvaluating clue {i}/{len(top_clues)}: {clue}")
            for n_guesses in range(1, 4):
                scores = np.array([
                    self.simulate_game(
                        clue, n_guesses, board_words,
                        good_words, bad_words, assassin_words, bystander_words
                    ) for _ in range(self.n_simulations)
                ])
                avg_score = np.mean(scores)
                print(f"  Guesses: {n_guesses}, Average score: {avg_score:.2f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_clue = clue
                    best_guesses = n_guesses
        
        print(f"\nBest clue found: {best_clue} with {best_guesses} guesses (score: {best_score:.2f})")
        return best_clue, best_guesses 