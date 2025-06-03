import numpy as np
MAX_VALUE = 100000

def Stfeat(feature, bf_list, enable_equal=True, enable_exceed=False):
    # idx_val = [abs(feature - bf) for bf in bf_list]
    # idx = np.argmin(idx_val)
    # return bf_list[idx]
    # if feature > max(bf_list):
    #     print("feature exceeded: ", feature)
    #     return -1
    if enable_exceed and feature > max(bf_list):
        return max(bf_list)
    if enable_equal:
        idx_val = [(bf - feature) if feature <= bf else MAX_VALUE for bf in bf_list]
        idx = np.argmin(idx_val)
        if idx_val[idx] == MAX_VALUE:
            print("feature exceeded: ", feature)
            print("bf_list: ", bf_list)
            assert False, "equal"
        return bf_list[idx]
    else:
        if feature >= max(bf_list):
            return feature
        idx_val = [(bf - feature) if feature < bf else MAX_VALUE for bf in bf_list]
        idx = np.argmin(idx_val)
        if idx_val[idx] == MAX_VALUE:
            print("feature exceeded: ", feature)
            print("bf_list: ", bf_list)
            assert False, "not equal"
        return bf_list[idx]
    

def calculate_p(t, i, j, bf_list, data):
    trans_succ_cnt = 0
    trans_total_cnt = 0
    for flow in data:
        if len(flow) < t + 2:
            continue
        x_t = flow[t]
        x_t_1 = flow[t + 1]
        if Stfeat(x_t, bf_list) == bf_list[i] and Stfeat(x_t_1, bf_list) == bf_list[j]:
            trans_succ_cnt += 1
        if Stfeat(x_t, bf_list) == bf_list[i]:
            trans_total_cnt += 1
    if trans_total_cnt == 0:
        return 0
    return trans_succ_cnt/trans_total_cnt


if __name__ == "__main__":
    bf_list = [0, 1, 2, 3, 4, 5]
    feature = 3
    print(Stfeat(feature, bf_list))
    feature = 6
    print(Stfeat(feature, bf_list))