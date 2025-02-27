import msgpack
import lzma
import numpy as np


def decode_numpy(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["__ndarray__"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj

with lzma.open("embeddings.msgpack.xz", "rb") as f:
    data = msgpack.unpackb(f.read(), object_hook=decode_numpy)

cosine_similarity = lambda a,b: np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


#testing similarities
word='text'
for i in data:
    simmy = cosine_similarity(data[word],data[i])
    if simmy>0.5:
        print(i,simmy)