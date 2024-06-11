import numpy as np
from itertools import combinations
from cvfgmc.utils import motion_model_refining

#np.random.seed(100)

max_row = 12
top_block = 13000
arr = np.random.randint(0, 256, (1000, 32400), dtype=np.uint8)

random_idx = np.random.choice(arr.shape[0], max_row, replace=False)
arr_selected = arr[random_idx, :]
top_block_idx = np.argpartition(np.min(arr_selected, axis=0), top_block)[:top_block]
arr_selected_sum = np.sum(np.min(arr_selected, axis=0)[top_block_idx])
print(random_idx, arr_selected_sum)

model_idx, mse_sum = motion_model_refining(arr, max_row, top_block)
#print(model_idx, mse_sum)


## Brute force all combinations
#minimum = 32400*1000
## Random select 12 rows
#for perm in combinations(range(arr.shape[0]), max_row):    
#    perm = np.array(perm)
#    perm_best = np.min(arr[perm, :], axis=0)
#    perm_best_idx = np.argpartition(perm_best, top_block)[:top_block]
#    perm_best_sum = np.sum(perm_best[perm_best_idx])
#    if perm_best_sum < minimum:
#        minimum = perm_best_sum
#        print(perm, perm_best_sum)
#        np.min(arr[perm, :], axis=0)
    
#for i in range(max_row):
#    print(arr[i, :])
#print(" ")

global_minimum = np.sum(arr)
global_minimum_idx = np.arange(max_row)
used_idx = np.arange(arr.shape[0])
while len(used_idx) > max_row:
    # Shuffle the rows
    np.random.shuffle(used_idx)
    used_arr = arr[used_idx, :] 
    #np.random.shuffle(used_arr)

    # Replace one by one
    minimum = 100000000
    ans_idx = np.arange(max_row)
    for i in range(max_row, used_idx.shape[0]):
        sub_minimum = minimum
        min_idx = None
        for j in range(max_row):
            new_idx = ans_idx.copy()
            new_idx[j] = i
            arr_sum = np.sum(np.min(used_arr[new_idx, :], axis=0))
            if arr_sum < sub_minimum:
                sub_minimum = arr_sum
                min_idx = new_idx
        if sub_minimum < minimum:
            ans_idx = min_idx
            minimum = sub_minimum
    print(len(used_idx), used_idx[ans_idx], minimum, global_minimum)

    # Merge the best rows
    current_best_arr = np.min(used_arr[ans_idx, :], axis=0)
    summed_mse = np.sum(current_best_arr)

    if summed_mse < global_minimum:
        global_minimum = summed_mse
        global_minimum_idx = used_idx[ans_idx]

    # Drop rows that
    smaller_ratio = []
    global_best_arr = np.min(arr[global_minimum_idx, :], axis=0)
    for i in range(used_arr.shape[0]):
        smaller_ratio.append(np.sum(used_arr[i] <= global_best_arr) / 13000)
    smaller_ratio = np.array(smaller_ratio)

    # Remove 10% of the worst
    remove_ratio = 0.01
    end_row_idx = int(np.ceil(remove_ratio * used_arr.shape[0]))
    remove_idx = np.argsort(smaller_ratio)[:end_row_idx]

    used_idx = np.delete(used_idx, remove_idx)

#print(global_minimum_idx, global_minimum)
best_rows = arr[global_minimum_idx, :]
merged_best_rows = np.min(best_rows, axis=0)
print(global_minimum_idx, np.sum(merged_best_rows))
# Print sum of top {top_block} in the merged_best_rows
top_block_idx = np.argpartition(merged_best_rows, top_block)[:top_block]
print(np.sum(merged_best_rows[top_block_idx]))