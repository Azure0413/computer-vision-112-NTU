import numpy as np

def hierarchical_b_order(start: int, end: int, step_size: int) -> list:
    '''
    return:
    -------
        order: list
            hierarchical-B order list. Each element is a list of 3 elements: [target, ref0, ref1]
    '''
    
    if (end - start) % step_size != 0:
        raise ValueError(f"Invalid step size. (end - start) must be divisible by step_size. Got: {end - start} % {step_size}")
    
    skip_target = [0 for _ in range(start, end+1)]

    order = [[start, None, None]]
    skip_target[start] = 1
    for i in range(start, end, step_size):
        order.append([i+step_size, None, None])
        _hierarchical_next_level(i, i + step_size, order)
        skip_target[i+step_size] = 1

    return np.array(order), np.array(skip_target)

def _hierarchical_next_level(srt: int, end: int, order: list):
    if srt + 1 == end:
        return
    mid = (srt + end) // 2
    order.append([mid, srt, end])
    _hierarchical_next_level(srt, mid, order)
    _hierarchical_next_level(mid, end, order)

if __name__ == "__main__":
    process_order = hierarchical_b_order(0, 128, 32)
    for i, target in enumerate(process_order):
        print(f"Order {i}: {target}")