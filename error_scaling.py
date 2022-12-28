import json
import math
import numpy as np
from datasize_scaling import svd, init_array

sizes = (10, 100, 1000, 5000)
rng = np.random.default_rng()


def rmse(svd_result, result) -> float:
    size = result.shape[0]
    square_error = 0.0
    for i in range(size):
        err = svd_result[i] - result[i]
        square_error += err * err

    return math.sqrt(square_error / size)


def get_all_svd_result(size : int, k : int) -> dict:
    #TODO : check svdvals order???
    svdvals_dict = dict()
    #svd
    init_data = init_array(size)
    _, svdvals_dict["svd"], _ = svd(init_data)
    _, svdvals_dict["svd+compressed"], _ = svd(init_data, n_svdvtriplets=k, is_compressed=True)
    #dask
    init_dask_data = init_array(size, is_dask=True)
    _, svdvals, _ = svd(init_dask_data, is_dask=True)
    _, svdvals_compressed, _ = svd(init_dask_data, n_svdvtriplets=k, is_dask=True, is_compressed=True)

    svdvals_dict["dask_svd"] = np.asarray(svdvals)
    svdvals_dict["dask_svd+compressed"] = np.asarray(svdvals_compressed)

    # sort in place
    # ref - https://stackoverflow.com/a/26984520
    np.sort(svdvals_dict["svd+compressed"][::-1])
    
    del init_data, init_dask_data

    return svdvals_dict

def run():
    n_svdvtriplets = 9
    result = {}
    for size in sizes:
        result[size] = {}
        print(f"current size: {size}")
        print(f"get all svdvals results")
        svdvals_dict = get_all_svd_result(size, n_svdvtriplets)
        print(f"calculating rmse")
        for key, svdvals in svdvals_dict.items():
            if key != "svd":
                result[size][key] = rmse(svdvals_dict["svd"], svdvals)
            if key.startswith("dask"):
                print(svdvals)

            print(f"finish processing key: {key}\n\n")

        del svdvals_dict

    result["compressed_size"] = n_svdvtriplets

    filename = "error.json"
    with open(filename, "w") as f:
        print(f"Saving results. \n\n")
        json.dump(result, f, indent=5)


if __name__ == '__main__':
    run()
