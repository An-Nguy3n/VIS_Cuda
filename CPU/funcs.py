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


@njit
def single_invest(WEIGHT,
                  INDEX,
                  PROFIT,
                  PROFIT_RANK,
                  PROFIT_RANK_NI,
                  INTEREST):
    size = INDEX.shape[0] - 1
    arr_inv_idx = np.zeros(size, np.int64)
    for i in range(size):
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end]
        arr_max = np.where(wgt_==max(wgt_))[0]
        if arr_max.shape[0] == 1:
            arr_inv_idx[i] = start + arr_max[0]
        else:
            arr_inv_idx[i] = -1

    """
    GeoMax, HarMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank
    """

    arr_profit = PROFIT[arr_inv_idx]
    arr_profit[arr_inv_idx==-1] = INTEREST
    GeoMax = geomean(arr_profit[1:])
    HarMax = harmean(arr_profit[1:])

    GeoLim = GeoMax
    HarLim = HarMax
    arr_inv_val = WEIGHT[arr_inv_idx]
    arr_inv_val[arr_inv_idx==-1] = 1.7976931348623157e+308
    arr_loop = arr_inv_val[1:]
    ValGeo = min(arr_loop)
    delta = max(1e-6*abs(ValGeo), 1e-6)
    ValGeo -= delta
    ValHar = ValGeo

    for v in arr_loop:
        temp_profit = np.where(arr_inv_val > v, arr_profit, INTEREST)
        geo = geomean(temp_profit[1:])
        har = harmean(temp_profit[1:])
        if geo > GeoLim:
            GeoLim = geo
            ValGeo = v

        if har > HarLim:
            HarLim = har
            ValHar = v

    arr_rank = PROFIT_RANK[arr_inv_idx]
    arr_rank = np.where(arr_inv_idx==-1, PROFIT_RANK_NI, arr_rank)
    GeoRank = geomean(arr_rank[1:])
    HarRank = harmean(arr_rank[1:])

    return GeoMax, HarMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank


@njit
def multi_invest_1(WEIGHT,
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

    ValGeoNgn = -1.0
    GeoNgn = -1.0
    ValHarNgn = -1.0
    HarNgn = -1.0
    temp_profit = np.zeros(size-1)
    for ii in range(len(arr_loop)):
        v = arr_loop[ii]
        bool_wgt = WEIGHT > v
        temp_profit[:] = 0.0
        for i in range(1, size):
            start, end = INDEX[i], INDEX[i+1]
            if np.count_nonzero(bool_wgt[start:end]) == 0:
                temp_profit[i-1] = INTEREST
            else:
                temp_profit[i-1] = PROFIT[start:end][bool_wgt[start:end]].mean()

        geo = geomean(temp_profit)
        har = harmean(temp_profit)
        if geo > GeoNgn:
            GeoNgn = geo
            ValGeoNgn = v

        if har > HarNgn:
            HarNgn = har
            ValHarNgn = v

    return ValGeoNgn, GeoNgn, ValHarNgn, HarNgn


@njit
def linear_regression(A, B):
    try:
        # Calculate means
        mean_A = np.mean(A)
        mean_B = np.mean(B)

        # Calculate covariance and variance
        cov_AB = 0.0
        for i in range(len(A)):
            cov_AB += (A[i]-mean_A)*(B[i]-mean_B)

        cov_AB /= len(A)
        var_A = np.var(A)

        # Estimate coefficients
        m = cov_AB / var_A
        b = mean_B - m * mean_A

        return m, b
    except:
        return 0.0, 0.0

@njit
def _find_slope(profit_, value_, y):
    if (value_ == 1.7976931348623157e+308).any() or y == 1.7976931348623157e+308:
        return 0.0, 0.0

    temp = np.argsort(value_)
    value = value_[temp]
    profit = profit_[temp]
    n = value.shape[0]
    arr_avg = np.zeros(n)
    for i in range(n):
        arr_avg[i] = np.mean(profit[i:])

    m1, b1 = linear_regression(value, arr_avg)
    slope_avg = m1*y + b1
    if np.isinf(slope_avg):
        slope_avg = 0.0

    if (value_ <= 0.0).any() or y <= 0.0:
        return slope_avg, 0.0

    arr_wgtavg = np.zeros(n)
    for i in range(n):
        arr_wgtavg[i] = np.sum(profit[i:] * value[i:]) / np.sum(value[i:])

    m2, b2 = linear_regression(value, arr_wgtavg)
    slope_wgtavg = m2*y + b2
    if np.isinf(slope_wgtavg):
        slope_wgtavg = 0.0

    return slope_avg, slope_wgtavg


@njit
def find_slope(WEIGHT, INDEX, PROFIT, INTEREST):
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size)
    arr_inv_value = np.zeros(size)
    for i in range(size-1, -1, -1):
        idx = size - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end]
        arr_max = np.where(wgt_==max(wgt_))[0]
        if arr_max.shape[0] == 1:
            arr_profit[idx] = PROFIT[arr_max[0]+start]
            arr_inv_value[idx] = wgt_[arr_max[0]]
        else:
            arr_profit[idx] = INTEREST
            arr_inv_value[idx] = 1.7976931348623157e+308

    temp_profit = arr_profit[:-1]
    temp_value = arr_inv_value[:-1]
    y = arr_inv_value[-1]
    slope_avg, slope_wgtavg = _find_slope(temp_profit, temp_value, y)
    return slope_avg, slope_wgtavg


@njit
def multi_invest_3(WEIGHT,
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

    ValGeoNgn3 = -1.0
    GeoNgn3 = -1.0
    ValHarNgn3 = -1.0
    HarNgn3 = -1.0
    temp_profit = np.zeros(size-3)
    for ii in range(len(arr_loop)):
        v = arr_loop[ii]
        bool_wgt = WEIGHT > v
        temp_profit[:] = 0.0
        reason = 0
        for i in range(size-3, 0, -1):
            start, end = INDEX[i], INDEX[i+1]
            inv_cyc_val = bool_wgt[start:end] & BOOL_ARG[start:end]
            if reason == 0:
                inv_cyc_sym = SYMBOL[start:end]
                end2, end3 = INDEX[i+2], INDEX[i+3]
                pre_cyc_val = bool_wgt[end:end2]
                pre_cyc_sym = SYMBOL[end:end2]
                pre2_cyc_val = bool_wgt[end2:end3]
                pre2_cyc_sym = SYMBOL[end2:end3]
                coms = np.intersect1d(pre_cyc_sym[pre_cyc_val], inv_cyc_sym[inv_cyc_val])
                coms = np.intersect1d(coms, pre2_cyc_sym[pre2_cyc_val])
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
        if geo > GeoNgn3:
            GeoNgn3 = geo
            ValGeoNgn3 = v

        if har > HarNgn3:
            HarNgn3 = har
            ValHarNgn3 = v

    return ValGeoNgn3, GeoNgn3, ValHarNgn3, HarNgn3
