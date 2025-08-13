import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
airfoil_file = pd.read_excel(rf'{base_dir}/Concrete_Data.xlsx')

from typing import Literal, Tuple
def return_cause_pair(relation: Literal["AB", "BA", "AB&BA"] = "AB", **kwargs):

    X = airfoil_file.iloc[:, :-1]
    y = airfoil_file.iloc[:, [-1]]

    if relation == "AB&BA":

        relation = ["AB", "BA"]
    else:
        relation = [relation]

    pair_data = []
    pair_name = []
    pair_cause = []

    for relation_ in relation:

        if relation_ == "AB":
            data_in_pair = [
                pd.concat([pd.DataFrame(airfoil_file.iloc[:, i]), pd.DataFrame(airfoil_file.iloc[:, -1])], axis=1)
                for i in range(airfoil_file.shape[1] - 1)]

        elif relation_ == "BA":

            data_in_pair = [
                pd.concat([pd.DataFrame(airfoil_file.iloc[:, -1]), pd.DataFrame(airfoil_file.iloc[:, i])], axis=1)
                for i in range(airfoil_file.shape[1] - 1)]

        else:
            raise AssertionError(f"return_cause_pair函数无法识别relation参数{relation_}")

        data_in_pair_format = []
        for pair in data_in_pair:
            pair.columns = [0, 1]
            data_in_pair_format.append(pair)

        pair_data.extend(data_in_pair_format)
        pair_name.extend([relation_ + "_" + f"[{str(col_name)}]" for col_name in airfoil_file.columns[:-1]])

        if relation_ == "AB":

            pair_cause.extend([1] * len(data_in_pair))


        else:
            pair_cause.extend([0] * len(data_in_pair))

    return pair_data, pair_name, pair_cause, X, y


if __name__ == '__main__':

    pair_values = return_cause_pair()