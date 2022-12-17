import dask.array as da
import scipy
import json
import numpy as np
from timeit import default_timer as timer

exponents = (1, 2, 3, 4, 5)
rng = np.random.default_rng()

def init_array(size : int, is_dask: bool = False):
    if not is_dask:
        return rng.standard_normal((size, size))
    else:
        return da.random.random(size=(size, size)).compute()

def svd(data, n_svdvtriplets: int = 10, is_dask: bool = False, is_compressed: bool = False):
    """
    Note that scipy.linalg.svds is not sorted yet
    Also not using random v0 vectir for svds
    """
    if not is_dask:
        if not is_compressed:
            _, _, _ = scipy.linalg.svd(data)
        else:
            _, _, _ = scipy.linalg.svds(
                data, k=n_svdvtriplets, which="LM"
            )
    else:
        if not is_compressed: 
            svdvec_left, svdvals, svdvec_right = da.linalg.svd(data)
        else:
            svdvec_left, svdvals, svdvec_right = da.linalg.svd_compressed(
            data, k=n_svdvtriplets
            ) 
        
        svdvec_left.compute()
        svdvals.compute()
        svdvec_right.compute()


def run(is_dask: bool = False, n_svdvtriplets: int = 0):
    is_compressed = False if n_svdvtriplets == 0 else True

    print(f"Is Dask -> {is_dask}")
    print(f"Is Compressed -> {is_compressed}")

    result = dict()
    for exp in exponents:
        size = pow(10, exp)
        print(f"current matrix size is {size} x {size}")
        array = init_array(size, is_dask)
        start = timer()
        svd(array, n_svdvtriplets)
        end = timer()

        t = end - start
        result[size] = t

        del array


    result["is_dask"] = is_dask
    result["is_compressed"] = is_compressed
    result["compressed_size"] = n_svdvtriplets

    filename = f"dask:{is_dask},compressed:{is_compressed}.json"
    with open(filename, "w") as f:
        print(f"Saving results. \n\n")
        json.dump(result, f, indent=5)


if __name__ == '__main__':
    run()
    run(True)
    run(n_svdvtriplets=10)
    run(True, n_svdvtriplets=10)
