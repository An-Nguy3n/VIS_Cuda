from numba import cuda
import math


@cuda.jit(device=True)
def max_of_array(arr, left, right, supremum):
    max_ = -1.7976931348623157e+308
    for i in range(left, right):
        if arr[i] < supremum and arr[i] > max_:
            max_ = arr[i]

    return max_


@cuda.jit(device=True)
def top_n_of_array(arr, left, right, result, start, n):
    supremum = 1.7976931348623157e+308
    for i in range(n):
        supremum = max_of_array(arr, left, right, supremum)
        result[start+i] = supremum


@cuda.jit(device=True)
def multi_invest_2(weight, threshold, t_idx, result,
                   INTEREST, INDEX, PROFIT, SYMBOL, BOOL_ARG):
    index_size = INDEX.shape[0]
    num_cycle = result.shape[0]

    reason = 0
    Geo2 = 0.0
    Har2 = 0.0
    for i in range(index_size-3, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        temp = 0.0
        count = 0
        check = False
        if reason == 0:
            end2 = INDEX[i+2]
            for k in range(start, end):
                if weight[k] > threshold and BOOL_ARG[k]:
                    check = True
                    sym = SYMBOL[k]
                    for s in range(end, end2):
                        if SYMBOL[s] == sym:
                            if weight[s] > threshold:
                                count += 1
                                temp += PROFIT[k]
                            break
        else:
            for k in range(start, end):
                if weight[k] > threshold and BOOL_ARG[k]:
                    check = True
                    count += 1
                    temp += PROFIT[k]

        if count == 0:
            Geo2 += math.log(INTEREST)
            Har2 += 1.0 / INTEREST
            if not check:
                reason = 1
        else:
            temp = temp / count
            Geo2 += math.log(temp)
            Har2 += 1.0 / temp
            reason = 0

        if i <= num_cycle and t_idx + 1 >= i:
            rs_idx = num_cycle - i
            n = float(index_size - 2 - i)
            result[rs_idx, 0] = math.exp(Geo2/n)
            result[rs_idx, 1] = n / Har2


@cuda.jit(device=True)
def maximum_min_3(weight, threshold, t_idx, result, temp_3,
                  INTEREST, INDEX, PROFIT):
    index_size = INDEX.shape[0]
    num_cycle = result.shape[0]
    minHar = 1.7976931348623157e+308
    minGeo = 1.7976931348623157e+308

    for i in range(index_size-2, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        temp = 0.0
        count = 0
        for k in range(start, end):
            if weight[k] > threshold:
                count += 1
                temp += PROFIT[k]

        if count == 0:
            temp_3[int(i%3)] = INTEREST
        else:
            temp_3[int(i%3)] = temp / count

        if i <= index_size - 4:
            Har = 0.0
            Geo = 0.0
            for j in range(3):
                Har += 1.0 / temp_3[j]
                Geo += math.log(temp_3[j])

            Har = 3.0 / Har
            Geo = math.exp(Geo/3.0)
            if Har < minHar: minHar = Har
            if Geo < minGeo: minGeo = Geo

        if i <= num_cycle and t_idx + 1 >= i:
            rs_idx = num_cycle - i
            result[rs_idx, 0] = minGeo
            result[rs_idx, 1] = minHar


@cuda.jit(device=True)
def multi_invest_3(weight, threshold, t_idx, result,
                   INTEREST, INDEX, PROFIT, SYMBOL, BOOL_ARG):
    index_size = INDEX.shape[0]
    num_cycle = result.shape[0]

    reason = 0
    Geo3 = 0.0
    Har3 = 0.0
    for i in range(index_size-4, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        temp = 0.0
        count = 0
        check = False
        if reason == 0:
            end2, end3 = INDEX[i+2], INDEX[i+3]
            for k in range(start, end):
                if weight[k] > threshold and BOOL_ARG[k]:
                    check = True
                    sym = SYMBOL[k]
                    pre_ = False
                    for s in range(end, end2):
                        if SYMBOL[s] == sym:
                            if weight[s] > threshold:
                                pre_ = True
                            break

                    if pre_:
                        for s in range(end2, end3):
                            if SYMBOL[s] == sym:
                                if weight[s] > threshold:
                                    count += 1
                                    temp += PROFIT[k]
                                break
        else:
            for k in range(start, end):
                if weight[k] > threshold and BOOL_ARG[k]:
                    check = True
                    count += 1
                    temp += PROFIT[k]

        if count == 0:
            Geo3 += math.log(INTEREST)
            Har3 += 1.0 / INTEREST
            if not check:
                reason = 1
        else:
            temp = temp / count
            Geo3 += math.log(temp)
            Har3 += 1.0 / temp
            reason = 0

        if i <= num_cycle and t_idx + 1 >= i:
            rs_idx = num_cycle - i
            n = float(index_size - 3 - i)
            result[rs_idx, 0] = math.exp(Geo3/n)
            result[rs_idx, 1] = n / Har3
