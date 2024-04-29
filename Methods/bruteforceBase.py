import numpy as np
from Methods.base import Base, operator_mapping
import queryFuncs as qf
import time
import pandas as pd
import os
import sqlite3
import evaluationFuncs as ef
import json
from numba import cuda
from datetime import datetime
import copy


def check_data_operands(op_name_1: dict, op_name_2: dict):
    if len(op_name_1) != len(op_name_2): return False

    op_1_keys = list(op_name_1.keys())
    op_2_keys = list(op_name_2.keys())
    for i in range(len(op_name_1)):
        if op_name_1[op_1_keys[i]] != op_name_2[op_2_keys[i]]:
            return False

    return True


def set_up(DB_PATH, DATA, LABEL, METHOD, LIST_FUNC, DIV_WGT_BY_MC):
    folder_data = f"{DB_PATH}/{LABEL}"
    os.makedirs(folder_data, exist_ok=True)

    if type(DATA) == str:
        data = pd.read_excel(DATA)
    else:
        data = DATA.copy()

    if DIV_WGT_BY_MC:
        MARKET_CAP = np.array(data.pop("MARKET_CAP"))
    else:
        MARKET_CAP = 1.0

    base = Base(data, 1.0, 0.0)
    if not os.path.exists(folder_data + "/operand_names.json"):
        with open(folder_data + "/operand_names.json", "w") as fp:
            json.dump(base.operand_name, fp, indent=4)
        operand_name = base.operand_name
    else:
        with open(folder_data + "/operand_names.json", "r") as fp:
            operand_name = json.load(fp)

    if not check_data_operands(base.operand_name, operand_name):
        raise Exception("Data đầu vào không trùng với data của lần chạy trước")

    folder_method = folder_data + f"/METHOD_{METHOD}"
    os.makedirs(folder_method, exist_ok=True)
    connection = sqlite3.connect(f"{folder_method}/f.db")

    list_field = []
    for key in LIST_FUNC:
        for v in ef.DESCRIPTION[key]:
            list_field.append((v, "REAL"))

    return data, connection, list_field, folder_method, MARKET_CAP


class BruteforceBase(Base):
    def __init__(self,
                 DB_PATH,
                 DATA_OR_PATH,
                 INTEREST,
                 VALUEARG_THRESHOLD,
                 NUM_CYCLE,
                 METHOD,
                 LIST_FUNC,
                 FILTERS,
                 DIV_WGT_BY_MC,
                 TARGET,
                 LABEL="auto",
                 TEMP_STORAGE_SIZE=10000,
                 MAX_NUM_OPR_PER_FML=100):
        print(FILTERS)
        if LABEL == "auto":
            LABEL = datetime.now().strftime("%Y%m%d_%H%M%S")
        data, connection, list_field, main_folder, MARKET_CAP = set_up(DB_PATH,
                                                                       DATA_OR_PATH,
                                                                       LABEL,
                                                                       METHOD,
                                                                       LIST_FUNC,
                                                                       DIV_WGT_BY_MC)
        super().__init__(data, INTEREST, VALUEARG_THRESHOLD)
        self.connection = connection
        self.list_func = LIST_FUNC
        self.list_field = list_field
        self.main_folder = main_folder
        self.MARKET_CAP = MARKET_CAP
        self.storage_size = TEMP_STORAGE_SIZE

        self.num_cycle = NUM_CYCLE
        self.target = TARGET
        self.filters = FILTERS

        self.cursor = connection.cursor()
        self.max_num_opr_per_fml = MAX_NUM_OPR_PER_FML
        self.time = time.time()

        self.temp_weights = np.zeros((
            self.storage_size+self.OPERAND.shape[0], self.OPERAND.shape[1]
        ), np.float64)
        self.temp_formulas = None
        self.count_temp = 0
        self.count_target = 0
        self.previos_count = 0

        self.d_INDEX = cuda.to_device(self.INDEX)
        self.d_PROFIT = cuda.to_device(self.PROFIT)
        self.d_SYMBOL = cuda.to_device(self.SYMBOL)
        self.d_BOOL_ARG = cuda.to_device(self.BOOL_ARG)

        getattr(self, f"prepare_{self.list_func[0]}")()

    def prepare_multi_invest_2(self):
        num_array = self.temp_weights.shape[0]
        num_threshold = 5*(self.INDEX.shape[0]-2)

        self.d_threshold = cuda.device_array((num_array, num_threshold), np.float64)
        self.d_results = cuda.device_array((num_array, num_threshold, self.num_cycle, 2), np.float64)
        self.d_finals = cuda.device_array((num_array, self.num_cycle, 4), np.float64)

    def add_to_temp_storage(self, weights, formulas):
        len_ = weights.shape[0]
        self.temp_weights[self.count_temp:self.count_temp+len_, :] = weights
        self.temp_formulas[self.count_temp:self.count_temp+len_, :] = formulas
        self.count_temp += len_

    def check_and_create_table(self, num_opr_per_fml):
        self.cursor.execute(qf.get_list_table())
        list_table = [t_[0] for t_ in self.cursor.fetchall()]
        max_cycle = self.data["TIME"].max()
        for cycle in range(max_cycle-self.num_cycle+1, max_cycle+1):
            if f"{cycle}_{num_opr_per_fml}" not in list_table:
                self.cursor.execute(qf.create_table(num_opr_per_fml, self.list_field, cycle))
                print("Create", f"{cycle}_{num_opr_per_fml}")
        self.connection.commit()

        num_array = self.temp_weights.shape[0]
        self.temp_formulas = np.zeros((num_array, 2*num_opr_per_fml), np.int16)
        self.d_encoded_fmls = cuda.device_array((num_array, num_opr_per_fml), np.int16)

    def change_checkpoint(self):
        self.start_id = ...
        self.cur_opr_per_fml = ...
        self.current = ...

    def save_history(self, flag=0):
        if self.previos_count != 0:
            cuda.synchronize()

            if self.list_func[0] == "multi_invest_2":
                encoded_fmls = self.old_d_encoded_fmls[:self.previos_count].copy_to_host()
                df_formula = pd.DataFrame(encoded_fmls)
                df_formula.columns = [f"E{k}" for k in range(self.old_cur_opr_per_fml)]
                df_formula.insert(loc=0, column="id", value=np.arange(self.old_start_id, self.old_start_id+self.previos_count))

                finals = self.d_finals[:self.previos_count].copy_to_host()
                list_df = []
                for i in range(self.num_cycle):
                    df = pd.DataFrame(finals[:, i, :])
                    df.columns = ["ValGeo2", "GeoNgn2", "ValHar2", "HarNgn2"]
                    for key, val in self.filters.items():
                        df = operator_mapping[val[0]](df, key, val[1])

                    list_df.append(df)

                max_cycle = self.data["TIME"].max()
                for i in range(self.num_cycle):
                    if len(list_df[i]) == 0: continue
                    # self.cursor.execute(qf.insert_rows(
                    #     f"{max_cycle-self.num_cycle+1+i}_{self.old_cur_opr_per_fml}",
                    #     df_formula, list_df[i]
                    # ))
                    df_index = list_df[i].index
                    df_save = pd.concat([df_formula.loc[df_index], list_df[i]], axis=1)
                    df_save.to_sql(
                        name=f"{max_cycle-self.num_cycle+1+i}_{self.old_cur_opr_per_fml}",
                        con=self.connection,
                        if_exists="append",
                        index=False,
                        method="multi"
                    )
                    self.count_target += len(list_df[i])

            self.old_start_id += self.previos_count
            self.change_checkpoint()
            np.save(self.main_folder+"/checkpoint.npy", np.asanyarray(self.history, dtype=object), allow_pickle=True)
            self.connection.commit()

            time_ = time.time()
            print("Saved", time_ - self.time, self.history)
            self.time = time_
            if self.count_target >= self.target:
                raise Exception("Đã sinh đủ số lượng")

        self.d_weights = cuda.to_device(self.temp_weights[:self.count_temp])
        self.d_formulas = cuda.to_device(self.temp_formulas[:self.count_temp])
        self.old_d_encoded_fmls = self.d_encoded_fmls

        if self.list_func[0] == "multi_invest_2":
            ef.multi_invest_2[ef.NUM_BLOCK, ef.BLOCK_DIM](
                self.d_weights,
                self.d_threshold,
                self.d_results,
                self.d_finals,
                self.INTEREST,
                self.d_INDEX,
                self.d_PROFIT,
                self.d_SYMBOL,
                self.d_BOOL_ARG,
                self.d_formulas,
                self.old_d_encoded_fmls,
                self.OPERAND.shape[0]
            )

        self.previos_count = self.count_temp
        self.old_start_id = self.start_id
        self.old_cur_opr_per_fml = self.cur_opr_per_fml
        self.start_id += self.count_temp
        self.count_temp = 0
        self.history = copy.deepcopy(self.current)

        if flag:
            self.save_history()
