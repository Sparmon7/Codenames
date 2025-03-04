import msgpack
import lzma
import numpy as np


def encode_numpy(obj):
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": obj.tobytes(),
            "dtype": str(obj.dtype),
            "shape": obj.shape
        }
    return obj

def decode_numpy(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


embeddings = {}
with open("glove.6B.200d.txt", "r", encoding="utf-8") as f:
    for line in f:
        vec = line.rstrip("\n").split(" ")
        if vec[0].isalpha() and not vec[0] is 'bulletinyyy':
            embeddings[vec[0]] = np.round(np.array(vec)[1:].astype(float),2)
        
print('saving')
with lzma.open("embeddings.msgpack.xz", "wb") as f:
    f.write(msgpack.packb(embeddings, default=encode_numpy))