from similar_images import Embeddings

def find_duplicates(array: np.ndarray, n:int=5):
    embeddings = Embeddings(array)
    pairs, distances = embeddings.duplicates(n=n)
    return pairs, distances

