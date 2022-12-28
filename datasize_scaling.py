import dask.array as da
import dask
import scipy
import scipy.sparse.linalg
import json
import numpy as np
from timeit import default_timer as timer

sizes = (10, 100, 1000)
rng = np.random.default_rng()
# improvement negligible
# dask.config.set({"optimization.fuse.ave-width": 5})

def init_array(size : int, is_dask: bool = False):
    if not is_dask:
        return rng.standard_normal((size, size))
    else:
        if size >= 5000:
            return da.random.random(size=(size, size), chunks=(size, 1000))
        return da.random.random(size=(size, size))

def svd(data, n_svdvtriplets: int = 10, is_dask: bool = False, is_compressed: bool = False, is_recompute: bool = False):
    """
    Note that scipy.linalg.svds is not sorted yet
    Also not using random v0 vectir for svds
    """
    if not is_dask:
        if not is_compressed:
            svdvec_left, svdvals, svdvec_right = scipy.linalg.svd(data)
        else:
            svdvec_left, svdvals, svdvec_right = scipy.sparse.linalg.svds(
                data, k=n_svdvtriplets, which="LM"
            )
    else:
        if not is_compressed: 
            svdvec_left, svdvals, svdvec_right = da.linalg.svd(data)
        else:
            svdvec_left, svdvals, svdvec_right = da.linalg.svd_compressed(
            data, k=n_svdvtriplets, compute=is_recompute
            ) 
        
        svdvec_left.compute()
        svdvals.compute()
        svdvec_right.compute()

    return svdvec_left, svdvals, svdvec_right


def run(is_dask: bool = False, n_svdvtriplets: int = 0, compute: bool=False):
    is_compressed = False if n_svdvtriplets == 0 else True

    print(f"Is Dask -> {is_dask}")
    print(f"Is Compressed -> {is_compressed}")

    result = dict()
    for size in sizes:
        print(f"current matrix size is {size} x {size}")
        array = init_array(size, is_dask)
        start = timer()
        svd(array, n_svdvtriplets, is_dask=is_dask, is_compressed=is_compressed, is_recompute=compute)
        end = timer()

        t = end - start
        result[size] = t

        del array


    result["is_dask"] = is_dask
    result["is_compressed"] = is_compressed
    result["compressed_size"] = n_svdvtriplets
    result["fuse.ave-width"] = 5

    filename = f"dask:{is_dask},compressed:{is_compressed},recompute:{compute}.json"
    with open(filename, "w") as f:
        print(f"Saving results. \n\n")
        json.dump(result, f, indent=5)


if __name__ == '__main__':
    # run(True, n_svdvtriplets=9)
    # run(True, n_svdvtriplets=9, compute=True)
    run(True, compute=True)
