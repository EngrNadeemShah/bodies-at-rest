import numpy as np
import chumpy as ch
import scipy.sparse

def print_dict(dict_: dict) -> None:
    """Prints a dictionary in a nice format."""
    print("\033[1m" + 'KEYS of the dictionary:' + "\033[0m")
    print(*dict_.keys(), sep = ' | ')
    print()

    for i, (k, v) in enumerate(dict_.items()):
        print("\033[1m" + f'{i} - K E Y -> {k}' + "\033[0m")
        if isinstance(v, np.ndarray):
            print(f'numpy array of shape: {v.shape}')
        elif isinstance(v, scipy.sparse.csc.csc_matrix):
            print(f'sparse matrix of shape: {v.shape}')
        elif isinstance(v, ch.Ch):
            print(f'chumpy array of shape: {v.shape}')
        elif isinstance(v, str):
            print(f'string: {v}')
        elif isinstance(v, dict):
            print(f'dictionary of keys: {v.keys()}')
        elif isinstance(v, list):
            print(f'list of length: {len(v)}')
        else:
            print(f'unknown type: {type(v)}')
        print()