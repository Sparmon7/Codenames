import msgpack
import lzma
import numpy as np
import time

similarity_cache = {}
MINIMUM_THRESHOLD = .45

# to load the data when utils is imported
print("Loading word embeddings...")
start_time = time.time()
with lzma.open("embeddings.msgpack.xz", "rb") as f:
    def decode_numpy(obj):
        if "__ndarray__" in obj:
            return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(obj["shape"])
        return obj
    data = msgpack.load(f, object_hook=decode_numpy)


def load_embeddings():
    """Return the pre-loaded embeddings."""
    return data


def load_single_embedding(word): # pro
    all_embeddings = load_embeddings()
    return all_embeddings.get(word, None)
    
def get_vocabulary():    
    return list(load_embeddings().keys())

def cosine_similarity(a, b):

    a_tuple = tuple(a.tolist())
    b_tuple = tuple(b.tolist())
    key = (a_tuple, b_tuple) if a_tuple < b_tuple else (b_tuple, a_tuple)

    if key not in similarity_cache:
        similarity_cache[key] = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    return similarity_cache[key]

def check_real_word(word):
    return word in data

def check_validity(word, board_words):
    return not any((word in bw) or (bw in word) for bw in board_words)

def check_minimum_threshold(word, good_words, threshold=MINIMUM_THRESHOLD):
    for i in good_words:
        if cosine_similarity(data[word], data[i]) > threshold:
            return True        
    return False

def remove_words_monte_carlo(good_words, bad_words, assassin_words, bystander_words, turn):
    board_words = [*good_words, *bad_words, *assassin_words, *bystander_words]
    # if turn:
    #     print('\nYour turn:')
    # else:
    #     print('\nOpponent\'s turn:')
    #     print(f"Remaining team words: {good_words}")

    # print(f"Remaining other team words: {bad_words}")
    # print(f"Remaining assassin word: {assassin_words}")
    # print(f"Remaining bystander words: {bystander_words}")
    # print('Type the words that were guessed, pressing enter after each word and enter when done')
    
    guessed_words = []
    while True:
        word = input().lower()
        if word == "":
            break
        if word in board_words:
            guessed_words.append(word)
            
            if word in good_words:
                good_words.remove(word)
            if word in bad_words:
                bad_words.remove(word)
            if word in assassin_words:
                assassin_words.remove(word)
            if word in bystander_words:
                bystander_words.remove(word)
        else:
            print("This word was not found, please try again")
    
    print(f"\nTurn complete. Guessed words: {guessed_words}")
    return good_words, bad_words, assassin_words, bystander_words