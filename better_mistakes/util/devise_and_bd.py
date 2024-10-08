import os
import lzma
import pickle
import numpy as np


def load_dict(opts):
    if opts.barzdenzler and opts.data in ["tiered-imagenet-84", "tiered-imagenet-224"]:
        fname = os.path.join(opts.data_dir, "tiered_imagenet_bd_embeddings.pkl.xz")
    elif opts.barzdenzler and opts.data in ["inaturalist19-84", "inaturalist19-224"]:
        fname = os.path.join(opts.data_dir, "inaturalist19_bd_embeddings.pkl.xz")
    elif opts.barzdenzler and opts.data in ["cifar100", "cifar-100"]:
        fname = os.path.join(opts.data_dir, "cifar100_embeddings.pkl.xz")
    elif opts.devise and opts.data in ["tiered-imagenet-84", "tiered-imagenet-224"]:
        fname = os.path.join(opts.data_dir, "tiered_imagenet_word2vec.pkl")
    else:
        raise ValueError(f"Unknown dataset {opts.data} for this method")

    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} is not a valid path.")
    if fname.endswith(".xz"):
        with lzma.open(fname, "rb") as f:
            return pickle.load(f)
    else:
        with open(fname, "rb") as f:
            return pickle.load(f)




def generate_sorted_embedding_tensor(opts):
    embedding_dict = load_dict(opts)
    matrix_len = len(embedding_dict)
    
    # Convert the first item to a numpy array to get the embedding length
    first_key = next(iter(embedding_dict))
    first_value = np.array(embedding_dict[first_key])
    emb_len = first_value.shape[0]
    
    matrix = np.zeros((matrix_len, emb_len))
    sorted_keys = sorted(embedding_dict.keys())
    
    for idx, name in enumerate(sorted_keys):
        matrix[idx] = np.array(embedding_dict[name])

    return matrix, sorted_keys
