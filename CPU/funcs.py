import numpy as np
from numba import njit


@njit
def geomean(arr):
    log_sum = sum(np.log(arr))
    return np.exp(log_sum/len(arr))

@njit
def harmean(arr):
    dnmntor = sum(1.0/arr)
    return len(arr)/dnmntor


@njit
def multi_invest_2(WEIGHT,
                   INDEX,
                   PROFIT,
                   SYMBOL,
                   INTEREST,
                   BOOL_ARG):
    size = INDEX.shape[0] - 1
    arr_loop = np.full((size-1)*5, -1.7976931348623157e+308, float)
    for i in range(1, size):
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = np.unique(WEIGHT[start:end])
        wgt_[::-1].sort()
        if len(wgt_) < 5:
            arr_loop[5*(i-1):5*(i-1)+len(wgt_)] = wgt_
        else:
            arr_loop[5*(i-1):5*i] = wgt_[:5]

    ValGeoNgn2 = -1.0
    GeoNgn2 = -1.0
    ValHarNgn2 = -1.0
    HarNgn2 = -1.0
    temp_profit = np.zeros(size-2)
    for ii in range(len(arr_loop)):
        v = arr_loop[ii]
        bool_wgt = WEIGHT > v
        temp_profit[:] = 0.0
        reason = 0
        for i in range(size-2, 0, -1):
            start, end = INDEX[i], INDEX[i+1]
            inv_cyc_val = bool_wgt[start:end] & BOOL_ARG[start:end]
            if reason == 0:
                inv_cyc_sym = SYMBOL[start:end]
                end2 = INDEX[i+2]
                pre_cyc_val = bool_wgt[end:end2]
                pre_cyc_sym = SYMBOL[end:end2]
                coms = np.intersect1d(pre_cyc_sym[pre_cyc_val], inv_cyc_sym[inv_cyc_val])
                isin = np.full(end-start, False)
                for j in range(end-start):
                    if inv_cyc_sym[j] in coms:
                        isin[j] = True
                lst_pro = PROFIT[start:end][isin]
            else:
                lst_pro = PROFIT[start:end][inv_cyc_val]

            if len(lst_pro) == 0:
                temp_profit[i-1] = INTEREST
                if np.count_nonzero(inv_cyc_val) == 0:
                    reason = 1
            else:
                temp_profit[i-1] = lst_pro.mean()
                reason = 0

        geo = geomean(temp_profit)
        har = harmean(temp_profit)
        if geo > GeoNgn2:
            GeoNgn2 = geo
            ValGeoNgn2 = v

        if har > HarNgn2:
            HarNgn2 = har
            ValHarNgn2 = v

    return ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2


@njit
def maximum_min_3(WEIGHT,
                  INDEX,
                  PROFIT,
                  INTEREST):
    size = INDEX.shape[0] - 1
    arr_loop = np.full((size-1)*5, -1.7976931348623157e+308, float)
    for i in range(1, size):
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = np.unique(WEIGHT[start:end])
        wgt_[::-1].sort()
        if len(wgt_) < 5:
            arr_loop[5*(i-1):5*(i-1)+len(wgt_)] = wgt_
        else:
            arr_loop[5*(i-1):5*i] = wgt_[:5]

    ValGeo3 = -1.0
    MinGeo3 = -1.0
    ValHar3 = -1.0
    MinHar3 = -1.0
    temp_profit = np.zeros(size-1)
    for ii in range(len(arr_loop)):
        v = arr_loop[ii]
        bool_wgt = WEIGHT > v
        temp_profit[:] = 0.0
        for i in range(size-1, 0, -1):
            start, end = INDEX[i], INDEX[i+1]
            inv_cyc_val = bool_wgt[start:end]
            lst_pro = PROFIT[start:end][inv_cyc_val]
            if len(lst_pro) == 0:
                temp_profit[i-1] = INTEREST
            else:
                temp_profit[i-1] = lst_pro.mean()

        CurGeo = 1.7976931348623157e+308
        CurHar = 1.7976931348623157e+308
        for k in range(size-3):
            tempGeo = geomean(temp_profit[k:k+3])
            tempHar = harmean(temp_profit[k:k+3])
            if tempGeo < CurGeo: CurGeo = tempGeo
            if tempHar < CurHar: CurHar = tempHar

        if CurGeo > MinGeo3:
            ValGeo3 = v
            MinGeo3 = CurGeo

        if CurHar > MinHar3:
            ValHar3 = v
            MinHar3 = CurHar

    return ValGeo3, MinGeo3, ValHar3, MinHar3
