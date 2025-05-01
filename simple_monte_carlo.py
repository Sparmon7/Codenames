import numpy as np
from utils import (
    load_embeddings,
    cosine_similarity,
    get_vocabulary,
    load_single_embedding,
)
import random
import gc
import logging
from datetime import datetime
import os
import sys

DEFAULT_N_SIMULATIONS = 100
DEFAULT_MAX_CANDIDATES = 10000
DEFAULT_LOG_FILE = "monte_carlo.log"
DEFAULT_CLEAR_LOG = True
DEFAULT_SIMILARITY_THRESHOLD = 0.3  
TOP_N_CANDIDATES = 10

class SimpleMonteCarlo:
    def __init__(
        self,
        n_simulations=DEFAULT_N_SIMULATIONS,
        max_candidates=DEFAULT_MAX_CANDIDATES,
        log_file=DEFAULT_LOG_FILE,
        clear_log=DEFAULT_CLEAR_LOG,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    ):
        print(f"\nInitializing Simple Monte Carlo with {n_simulations} simulations...")
        self.n_simulations = n_simulations
        self.max_candidates = max_candidates
        self.similarity_threshold = similarity_threshold
        self.data = load_embeddings()
        self._loaded_embeddings = set()
        self.similarities_buffer = None  # reusing similarities

        # Set up logging
        if clear_log and os.path.exists(log_file):
            os.remove(log_file)

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(message)s",
            filemode="a",  # append mode
        )
        self.logger = logging.getLogger("MonteCarlo")

        # Add run demarcation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"NEW RUN - {timestamp}")
        self.logger.info("=" * 80 + "\n")

    def print_progress(self, current, total):
        """used chat to make this progress bar thingy"""
        update_interval = max(1, total // 20)  
        if current % update_interval == 0 or current == total:
            dots = "." * (current // update_interval % 4)  # Cycle through 0-3 dots
            sys.stdout.write(f"\rEvaluating{dots}{' ' * (3 - len(dots))}")  # Clear dots
            sys.stdout.flush()
            if current == total:
                print()  # New line when complete

    def get_embedding(self, word):
        if word not in self.data:
            self.data[word] = load_single_embedding(word)
            self._loaded_embeddings.add(word)   
        return self.data[word]

    def get_word_similarities(self, clue, board_words):
        return np.array(
            [
                cosine_similarity(self.data[clue], self.data[word])
                for word in board_words
            ]
        )

    def sample_guesses(self, similarities, n_guesses):
        # softmax
        exp_sims = np.exp(similarities)
        probs = exp_sims / np.sum(exp_sims)

        # sample without replacement
        return np.random.choice(
            len(probs), size=min(n_guesses, len(probs)), replace=False, p=probs
        )

    def simulate_opponent_guesses(
        self, clue, board_words, good_words, bad_words, n_guesses=2
    ):
        """Simulate opponent's guesses based on word similarities."""
        similarities = self.get_word_similarities(clue, board_words)
        guesses = self.sample_guesses(similarities, n_guesses)

        # Count good and bad guesses
        good_guesses = sum(1 for guess in guesses if board_words[guess] in good_words)
        bad_guesses = sum(1 for guess in guesses if board_words[guess] in bad_words)
        return good_guesses, bad_guesses

    def evaluate_clue(self, clue, board_words, good_words, bad_words, n_guesses):
        """Evaluate a clue by simulating opponent guesses multiple times."""
        total_good_guesses = 0
        total_bad_guesses = 0

        for _ in range(self.n_simulations):
            good_guesses, bad_guesses = self.simulate_opponent_guesses(
                clue, board_words, good_words, bad_words, n_guesses
            )
            total_good_guesses += good_guesses
            total_bad_guesses += bad_guesses

        # Score is based on good guesses minus bad guesses
        avg_good_guesses = total_good_guesses / self.n_simulations
        avg_bad_guesses = total_bad_guesses / self.n_simulations
        return avg_good_guesses - avg_bad_guesses

    def get_candidate_clues(self, board_words, good_words, bad_words):
        """
        Generate a list of candidate clues by filtering and scoring them based on their similarity to board words.

        - Sample a subset of words from the vocabulary for an intial pool.
        - maximizing similarity to 'good' words and minimizing similarity to 'bad' words.
        - calculate score average sim to good words - average sim to bad words.
        - select  top_n candidates based on their scores.
        """
        # Get initial candidate pool
        vocabulary = get_vocabulary()
        candidate_pool = random.sample(
            vocabulary, min(self.max_candidates, len(vocabulary))
        )
        self.logger.info(
            f"Sampled {len(candidate_pool)} initial candidates from vocabulary"
        )

        candidates = []
        total_candidates = len(candidate_pool)

        # First pass: get clues with high similarity to good words
        for i, clue in enumerate(candidate_pool, 1):
            self.print_progress(i, total_candidates)

            if not any(clue in word or word in clue for word in board_words):
                similarities = self.get_word_similarities(clue, board_words)

                # Calculate scores for good and bad words
                good_scores = similarities[: len(good_words)]
                bad_scores = similarities[
                    len(good_words) : len(good_words) + len(bad_words)
                ]

                # Require minimum similarity to at least one good word
                if np.max(good_scores) < self.similarity_threshold:  
                    continue

                # Score = difference between good and bad similarities
                good_score = np.mean(good_scores)
                bad_score = np.mean(bad_scores)
                score = good_score - bad_score

                candidates.append((clue, score, good_scores))

        # Sort and take top_n candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:TOP_N_CANDIDATES]

    def generate_best_clue(
        self, good_words, bad_words, assassin_words, bystander_words
    ):
        print("\nGenerating best clue...")
        board_words = [*good_words, *bad_words, *assassin_words, *bystander_words]

        self.logger.info("\nBoard State:")
        self.logger.info(f"Good words: {good_words}")
        self.logger.info(f"Bad words: {bad_words}")
        self.logger.info(f"Assassin words: {assassin_words}")
        self.logger.info(f"Bystander words: {bystander_words}\n")

        candidates = self.get_candidate_clues(board_words, good_words, bad_words)
        self.logger.info(f"Found {len(candidates)} candidate clues")

        best_clue = None
        best_score = -float("inf")
        best_guesses = 0

        for i, (clue, _, _) in enumerate(candidates, 1):
            self.logger.info(f"\nEvaluating clue {i}/{len(candidates)}: {clue}")

            # Try different numbers of guesses
            for n_guesses in range(1, 4):
                score = self.evaluate_clue(
                    clue, board_words, good_words, bad_words, n_guesses
                )
                self.logger.info(f"  Guesses: {n_guesses}, Score: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_clue = clue
                    best_guesses = n_guesses

        self.logger.info(
            f"\nBest clue found: {best_clue} with {best_guesses} guesses (score: {best_score:.2f})"
        )
        self.logger.info("\n" + "=" * 80 + "\n")  # demaraction for the log file
        print(
            f"\nBest clue found: {best_clue} with {best_guesses} guesses (score: {best_score:.2f})"
        )
        indices=sorted(range(len(good_words)), key=lambda i: self.get_word_similarities(best_clue,good_words)[i], reverse=True)[:best_guesses]
        
        temp = self.get_word_similarities(best_clue,good_words)
        for i in range(len(good_words)):
            print(temp[i], good_words[i])
        return best_clue, best_guesses, [good_words[i] for i in indices] 

    def cleanup(self):
        """Free memory explicitly"""
        self.data = None
        self.similarities_buffer = None
        self._loaded_embeddings = None
        gc.collect()

