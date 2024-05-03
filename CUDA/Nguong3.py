from numba import cuda
from CUDA import deviceFuncs as devF
import math

NUM_BLOCK = 300
BLOCK_DIM = 64


@cuda.jit
def multi_invest_3(weights, thresholds, results, finals,
                    INTEREST, INDEX, PROFIT, SYMBOL, BOOL_ARG,
                    formulas, encoded_fmls, num_operand):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    g = cuda.cg.this_grid()
    thread_num_ = bx*BLOCK_DIM + tx

    num_array = weights.shape[0]
    index_size = INDEX.shape[0]
    temp = index_size - 2
    num_threshold = 5*temp
    num_cycle = results.shape[2]
    num_opr_per_fml = encoded_fmls.shape[1]

    # Thresholds
    thread_num = thread_num_
    while thread_num < num_array * temp:
        ix = int(thread_num % temp)
        iy = int(math.floor(thread_num/temp))
        devF.top_n_of_array(
            weights[iy], INDEX[ix+1], INDEX[ix+2], thresholds[iy], ix*5, 5
        )
        thread_num += NUM_BLOCK * BLOCK_DIM
    g.sync()

    # Results
    thread_num = thread_num_
    while thread_num < num_array * num_threshold:
        ix = int(thread_num % num_threshold)
        iy = int(math.floor(thread_num/num_threshold))
        devF.multi_invest_3(
            weights[iy], thresholds[iy, ix], math.floor(ix/5), results[iy, ix],
            INTEREST, INDEX, PROFIT, SYMBOL, BOOL_ARG
        )
        thread_num += NUM_BLOCK * BLOCK_DIM
    g.sync()

    # Finals
    thread_num = thread_num_
    while thread_num < num_array * 2*num_cycle:
        iz = int(thread_num % 2)
        ix = int((thread_num//2) % num_cycle)
        iy = int(math.floor((thread_num//2)/num_cycle))

        result = results[iy]
        threshold = thresholds[iy]
        final = finals[iy]

        final[ix, 2*iz] = threshold[0]
        final[ix, 2*iz+1] = result[0, ix, iz]
        for i in range(1, num_threshold):
            if result[i, ix, iz] > final[ix, 2*iz+1]:
                final[ix, 2*iz] = threshold[i]
                final[ix, 2*iz+1] = result[i, ix, iz]
        thread_num += NUM_BLOCK * BLOCK_DIM

    # Formula
    thread_num = thread_num_
    while thread_num < num_array * num_opr_per_fml:
        ix = int(thread_num % num_opr_per_fml)
        iy = int(math.floor(thread_num/num_opr_per_fml))
        encoded_fmls[iy, ix] = formulas[iy, 2*ix]*num_operand + formulas[iy, 2*ix+1]
        thread_num += NUM_BLOCK * BLOCK_DIM
