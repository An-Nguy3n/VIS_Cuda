def get_list_table():
    return '''SELECT name FROM sqlite_master WHERE type = "table";'''


def create_table(len_formula, list_field, cycle):
    list_formula_col = [f'"E{i}" INTEGER NOT NULL,' for i in range(len_formula)]
    list_field_col = [f'"{field[0]}" {field[1]},' for field in list_field]
    temp = "\n    "
    return f'''CREATE TABLE "{cycle}_{len_formula}" (
    "id" INTEGER NOT NULL,
    {temp.join(list_formula_col)}
    {temp.join(list_field_col)}
    PRIMARY KEY ("id")
)'''


def insert_rows(table_name, dff, dfv):
    temp = "".join([
        "(" + ",".join(
            [str(v1) for v1 in dff.loc[i]] + [str(v2) for v2 in dfv.loc[i]]
        ) + ")," for i in dfv.index
    ])[:-1]
    return f'''INSERT INTO "{table_name}" VALUES {temp};'''
