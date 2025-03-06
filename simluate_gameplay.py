import msgpack
import lzma
import numpy as np
import random
from multiprocessing import Pool, cpu_count
import time
import cProfile

# Adjustable parameters
skill_level = 21
assassin_penalty = -9

# Cache for embeddings
board_embeddings_cache = {}


def decode_numpy(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(
            obj["shape"]
        )
    return obj


# Load data once globally
print("Loading embeddings...")
start_time = time.time()

with lzma.open("embeddings.msgpack.xz", "rb") as f:
    data = msgpack.unpackb(f.read(), object_hook=decode_numpy)

print(f"Embeddings loaded in {time.time() - start_time:.2f} seconds")

# Load words once globally
print("Loading words...")
words = open("words.txt").read().replace("\n", " ").split()
words = [w.lower() for w in words]
print(f"Loaded {len(words)} words")


def calculate_expected_dp(softmaxes, guesses):
    dp = np.zeros((guesses + 1, 18))

    for i in range(18):
        if i < 9:  # team words
            dp[1, i] = softmaxes[i]
        elif i < 17:  # opponent words
            dp[1, i] = -softmaxes[i]
        elif i == 17:  # assassin
            dp[1, i] = assassin_penalty * softmaxes[i]

    # fill dp table for multiple guesses
    for g in range(2, guesses + 1):
        for i in range(9):  # Only process team words for multiple guesses
            prob = softmaxes[i]

            next_probs = softmaxes.copy()
            next_probs[i] = 0
            total = next_probs.sum()
            if total > 0:
                next_probs /= total

            dp[g, i] = prob * (1 + sum(next_probs[j] * dp[g - 1, j] for j in range(18)))

    return dp


def calculate_expected_rec(softmaxes, guesses):
    dp = np.zeros((guesses + 1, 18))

    # intialize for recursive calculations
    for word in range(18):
        for g in range(1, guesses + 1):
            dp[g, word] = calculate_expected_word_rec(
                word, softmaxes, g, assassin_penalty
            )

    return dp


def calculate_expected_word_rec(word, softmaxes, guesses, assassin_penalty):
    temp = softmaxes.copy()

    if word < 9:  # team words
        if guesses == 1:
            return temp[word]
        else:
            # for multiple guesses
            prob = temp[word]

            # zero out probability for this word and renormalize
            temp[word] = 0
            total = temp.sum()
            if total > 0:
                temp = temp / total

            expected_words = prob  # base value for correctly guessing this word

            for w in range(18):
                if temp[w] > 0:
                    expected_words += prob * calculate_expected_word_rec(
                        w, temp, guesses - 1, assassin_penalty
                    )

            return expected_words
    elif word >= 9 and word < 17:  # opponent words
        return -1 * temp[word]
    elif word == 17:  # assassin
        return assassin_penalty * temp[word]
    else:  # bystander
        return 0


def compute_cosines_batch(candidate_vectors, board_matrix):
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    board_norms = np.linalg.norm(board_matrix, axis=1, keepdims=True)

    dot_products = np.dot(candidate_vectors, board_matrix.T)

    norms_outer = np.dot(candidate_norms, board_norms.T)
    return dot_products / norms_outer


def get_board_embeddings(board_words):
    embeddings = []
    for word in board_words:
        if word not in board_embeddings_cache:
            board_embeddings_cache[word] = data[word]
        embeddings.append(board_embeddings_cache[word])
    return np.vstack(embeddings)


def check_validity(clue, board_words):
    return all((clue not in word and word not in clue) for word in board_words)


def get_valid_clues(board_words, sample_size=5000):
    # randomly sampling
    candidates = random.sample(list(data.keys()), min(sample_size, len(data)))
    return [clue for clue in candidates if check_validity(clue, board_words)]


def select_best_clue_batch(
    board_words, board_matrix, board_norms, valid_clues, batch_size=500
):
    """Select the best clue by processing candidates in batches"""
    max_expected = -float("inf")
    best_clue = None
    best_softmaxes = None
    best_guesses = None

    for i in range(0, len(valid_clues), batch_size):
        batch_clues = valid_clues[i : i + batch_size]

        batch_vectors = np.vstack([data[clue] for clue in batch_clues])
        all_cosines = compute_cosines_batch(batch_vectors, board_matrix)

        # Process each candidate
        for idx, clue in enumerate(batch_clues):
            cosines = all_cosines[idx]

            # potential check - sum of cosines for team words
            potential = np.sum(cosines[:9])
            if potential < max_expected / 3:  # Skip low potential clues
                continue

            exp_values = np.exp(skill_level * cosines)
            softmaxes = exp_values / exp_values.sum()

            # evaluate for different numbers of guesses
            for guesses in range(1, 4):
                dp_table = calculate_expected_rec(softmaxes, guesses)
                expected_total = sum(dp_table[guesses, i] for i in range(18))

                if expected_total > max_expected:
                    max_expected = expected_total
                    best_clue = clue
                    best_softmaxes = softmaxes
                    best_guesses = guesses

    return best_clue, best_guesses, best_softmaxes


def simulate_turn(
    board_words,
    remaining_team_indices,
    board_matrix,
    board_norms,
    valid_clues,
    verbose=False,
):
    clue, guesses, softmaxes = select_best_clue_batch(
        board_words, board_matrix, board_norms, valid_clues
    )

    if verbose:
        print(
            f"Chosen clue: {clue}, with {guesses} guess(es) and softmaxes: {softmaxes}"
        )

    if clue is None:
        return 0, remaining_team_indices  # no valid clue found

    available_indices = list(range(18))  # only consider first 18 words

    # remove words already guessed from team words in available_indices
    for idx in range(9):
        if idx not in remaining_team_indices:
            if idx in available_indices:
                available_indices.remove(idx)

    turn_guesses = 0

    for _ in range(guesses):
        if not available_indices:
            break

        # renormalize probabilities for available words
        probs = np.array([softmaxes[i] for i in available_indices])
        probs = probs / probs.sum()

        # sample word
        chosen_idx_pos = np.random.choice(len(available_indices), p=probs)
        chosen_idx = available_indices[chosen_idx_pos]
        turn_guesses += 1

        if verbose:
            guessed_word = board_words[chosen_idx]
            print(
                f"Turn guess {turn_guesses}: guessed '{guessed_word}' (index {chosen_idx})"
            )

        # check if it's a team word
        if chosen_idx < 9 and chosen_idx in remaining_team_indices:
            remaining_team_indices.remove(chosen_idx)
            available_indices.remove(chosen_idx)
        else:
            # break turn if wrong
            if verbose:
                print(f"Incorrect guess. Ending turn.")
            break

    return turn_guesses, remaining_team_indices


def simulate_game(seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    board_words = random.sample(words, 25)
    remaining_team_indices = list(range(9))  # 0-8 are team words

    if verbose:
        print("Initial board words:")
        for idx, word in enumerate(board_words):
            marker = "TEAM" if idx < 9 else "OTHER"
            print(f"{idx:2}: {word} ({marker})")

    board_matrix = get_board_embeddings(board_words)
    board_norms = np.linalg.norm(board_matrix, axis=1)

    valid_clues = get_valid_clues(board_words)

    game_turns = 0
    while remaining_team_indices:
        game_turns += 1
        if verbose:
            print(f"\n--- Turn {game_turns} ---")
            print(f"Remaining team indices: {remaining_team_indices}")
        if game_turns > 20:  # safety break
            if verbose:
                print("Exceeded maximum turns. Ending game simulation.")
            break

        turn_guesses, remaining_team_indices = simulate_turn(
            board_words,
            remaining_team_indices,
            board_matrix,
            board_norms,
            valid_clues,
            verbose=verbose,
        )
        if verbose:
            print(
                f"Turn ended with {turn_guesses} guesses. Remaining team indices: {remaining_team_indices}"
            )

    if verbose:
        print(f"\nGame completed in {game_turns} turns.")

    return game_turns


def run_simulations_parallel(n_games=100):
    print(
        f"Running {n_games} simulations {('using ' + cpu_count() + 'processes...') if n_games > 1 else ''}"
    )
    start_time = time.time()

    seeds = [random.randint(0, 10000) for _ in range(n_games)]

    if n_games == 1:
        results = simulate_game(seeds[0], verbose=True)
    else:
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(simulate_game, seeds)

    avg_turns = sum(results) / len(results)
    print(f"Average number of turns: {avg_turns:.2f}")
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")

    # Distribution of turns
    turns_count = {}
    for turns in results:
        turns_count[turns] = turns_count.get(turns, 0) + 1

    print("\nDistribution of turns:")
    for turns in sorted(turns_count.keys()):
        percentage = turns_count[turns] / n_games * 100
        print(f"{turns} turns: {turns_count[turns]} games ({percentage:.1f}%)")

    return avg_turns


if __name__ == "__main__":
    run_simulations_parallel(1)
