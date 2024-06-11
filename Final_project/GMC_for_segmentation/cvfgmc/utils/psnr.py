import numpy as np
from concurrent.futures import ProcessPoolExecutor 
from .frames import gridding
import numba as nb


def _mse_last_2d(target: np.ndarray, truth: np.ndarray) -> np.ndarray:
    '''
    params:
    -------
        target: np.ndarray
            (..., block_size, block_size) size numpy array
        truth: np.ndarray
            (..., block_size, block_size) size numpy array
    return:
    -------
        mse: 
            (..., ) mean squared error
    '''
    return np.mean((target.astype(np.int32) - truth.astype(np.int32))**2, axis=(-2, -1))

@nb.njit(parallel=True)
def _mse_last_2d_numba(target: np.ndarray, truth: np.ndarray) -> np.ndarray:
    '''
    params:
    -------
        target: np.ndarray
            (..., block_size, block_size) size numpy array
        truth: np.ndarray
            (..., block_size, block_size) size numpy array
    return:
    -------
        mse: 
            (..., ) mean squared error
    '''
    result_shape = target.shape[:-2]
    mse = np.zeros(result_shape, dtype=np.float64)

    for i in nb.prange(target.shape[0]):
        for index in np.ndindex(target.shape[1:-2]):
            full_index = (i, *index)
            sub_target = target[full_index]
            sub_truth = truth[full_index]
            total_error = 0.0
            count = sub_target.size
            
            for x in range(sub_target.shape[0]):
                for y in range(sub_target.shape[1]):
                    total_error += (sub_target[x, y] - sub_truth[x, y])**2
            
            mse[full_index] = total_error / count

    return mse


def evaluate(target: np.ndarray, truth: np.ndarray, block_size: list, top: int=13000) -> float:
    '''
    params:
    -------
        target: np.ndarray
            (N, Height, Width) size numpy array
        truth: np.ndarray
            (N, Height, Width) size numpy array
        top: int
            Top k blocks to consider
    return:
    -------
        psnr: 
            (N, ) peak signal to noise ratio
        
    '''
    if target.shape != truth.shape:
        # If truth can be broadcasted to target, then repeat truth to match target
        if target.shape[1:] != truth.shape:
            raise ValueError(f"target and truth must have same shape. Or truth can be broadcasted to target. Got {target.shape} and {truth.shape}")
        else:
            truth = np.repeat(truth[np.newaxis], target.shape[0], axis=0)
    
    if target.ndim != 3:
        if target.ndim == 2:
            target = target.reshape(1, target.shape[0], target.shape[1])
            truth = truth.reshape(1, truth.shape[0], truth.shape[1])
        else:
            raise ValueError(f"target and truth must have 2 dimensions. Got {target.ndim} and {truth.ndim}")

    target = gridding(target, block_size)   # (129, 135, 240, 16, 16)
    truth = gridding(truth, block_size)     # (129, 135, 240, 16, 16)

    mse_block_grid = _mse_last_2d_numba(target, truth)
    mse_block_flat = mse_block_grid.reshape(-1, mse_block_grid.shape[-2]*mse_block_grid.shape[-1])  # (129, 32400)
    mse_block_top_k_idx = np.argpartition(mse_block_flat, top, axis=-1)[:, :top]    # (129, 13000)

    # Construct a bool mask for the top k blocks
    mse_block_mask = np.zeros_like(mse_block_flat, dtype=bool)
    mse_block_mask[np.arange(mse_block_mask.shape[0])[:, None], mse_block_top_k_idx] = True
    mse_block_mask = mse_block_mask.reshape(mse_block_grid.shape)

    mse_selected = np.where(mse_block_mask, mse_block_grid, 0)
    mse = np.sum(mse_selected, axis=(-2, -1)) / top
    psnr = 10 * np.log10(255**2 / mse)

    if target.shape[0] == 1:
        psnr = psnr[0]
        mse_block_mask = mse_block_mask[0]
        mse_block_grid = mse_block_grid[0]

    return psnr, mse_block_mask.astype(np.uint8), mse_block_grid


def motion_model_refining(model_block_mse: np.ndarray, top_k_model, top_p_block=13000):
    '''
    params:
    -------
        model_block_mse: np.ndarray
            (M, Height, Width) size numpy array. M is the number of Motion models. Unit of Height and Width is block.
        top_k: int
            number of top k blocks to consider
    '''
    mse_block_flat = model_block_mse.reshape(model_block_mse.shape[0], -1)  # (M, 32400)

    # We iterating through the remaining models and find the best k models
    minimum = np.iinfo(np.int32).max
    chosen_model_idx = np.arange(top_k_model)
    for j in np.arange(top_k_model, mse_block_flat.shape[0]):
        sub_minimum = minimum
        sub_chosen_model_idx = None
        for k in np.arange(top_k_model):
            new_chosen_model_idx = chosen_model_idx.copy()
            new_chosen_model_idx[k] = j

            min_mse = np.min(mse_block_flat[new_chosen_model_idx, :], axis=0)
            top_p_block_idx = np.argpartition(min_mse, top_p_block, axis=-1)[:top_p_block]
            mse_sum = np.sum(min_mse[top_p_block_idx])

            if mse_sum < sub_minimum:
                sub_minimum = mse_sum
                sub_chosen_model_idx = new_chosen_model_idx

        if sub_minimum < minimum:
            chosen_model_idx = sub_chosen_model_idx
            minimum = sub_minimum
    
    current_best_mse = np.min(mse_block_flat[chosen_model_idx, :], axis=0)
    top_p_block_idx = np.argpartition(current_best_mse, top_p_block, axis=-1)[:top_p_block]
    mse_sum = np.sum(current_best_mse[top_p_block_idx])

    return chosen_model_idx, mse_sum

def motion_model_refining_accurate(model_block_mse: np.ndarray, top_k_model, top_p_block=13000, decay_rate=0.9):
    '''
    params:
    -------
        model_block_mse: np.ndarray
            (M, Height, Width) size numpy array. M is the number of Motion models. Unit of Height and Width is block.
        top_k: int
            number of top k blocks to consider
    '''
    mse_block_flat = model_block_mse.reshape(model_block_mse.shape[0], -1).copy()  # (M, 32400)

    global_minimum = np.iinfo(np.int64).max
    global_minimum_idx = np.arange(top_k_model)
    remaining_model_idx = np.arange(mse_block_flat.shape[0])
    while len(remaining_model_idx) > top_k_model:
        # Shuffle the remaining models
        np.random.shuffle(remaining_model_idx)
        remaining_model = mse_block_flat[remaining_model_idx, :]
        # We iterating through the remaining models and find the best k models
        minimum = np.iinfo(np.int64).max
        chosen_model_idx = np.arange(top_k_model)
        for j in range(top_k_model, remaining_model_idx.shape[0]):
            sub_minimum = minimum
            sub_chosen_model_idx = None
            for k in range(top_k_model):
                new_chosen_model_idx = chosen_model_idx.copy()
                new_chosen_model_idx[k] = j

                min_mse = np.min(remaining_model[new_chosen_model_idx, :], axis=0)
                top_p_block_idx = np.argpartition(min_mse, top_p_block, axis=-1)[:top_p_block]
                mse_sum = np.sum(min_mse[top_p_block_idx])

                if mse_sum < sub_minimum:
                    sub_minimum = mse_sum
                    sub_chosen_model_idx = new_chosen_model_idx

            if sub_minimum < minimum:
                chosen_model_idx = sub_chosen_model_idx
                minimum = sub_minimum

        # Merge the best rows
        current_best_mse = np.min(remaining_model[chosen_model_idx, :], axis=0)
        top_p_block_idx = np.argpartition(current_best_mse, top_p_block, axis=-1)[:top_p_block]
        current_best_summed_mse = np.sum(current_best_mse[top_p_block_idx])

        if current_best_summed_mse < global_minimum:
            global_minimum = current_best_summed_mse
            global_minimum_idx = remaining_model_idx[chosen_model_idx]

        # Drop rows based on decay rate
        bad_model = []
        global_best_arr = np.min(mse_block_flat[global_minimum_idx, :], axis=0)
        for i in range(remaining_model_idx.shape[0]):
            bad_model.append(np.sum(remaining_model[i] <= global_best_arr) / top_p_block)
        bad_model = np.array(bad_model)

        # Remove the worst model
        end_model_idx = int(np.ceil((1 - decay_rate) * remaining_model_idx.shape[0]))
        remove_model_idx = np.argsort(bad_model)[:end_model_idx]

        remaining_model_idx = np.delete(remaining_model_idx, remove_model_idx)

    top_p_block_idx = np.argpartition(current_best_mse, top_p_block, axis=-1)[:top_p_block]
    mse_sum = np.sum(current_best_mse[top_p_block_idx])

    return global_minimum_idx, global_minimum