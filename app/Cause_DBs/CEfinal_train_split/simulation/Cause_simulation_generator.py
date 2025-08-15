import pandas as pd

from Projects.Muti_func_creat.muti_func_test import gen_y_exp


def gen_linear(toward='AB', seed=0, num=100):
    for i in range(num):

        x_, X_, y_exp_, x_picked_ = gen_y_exp(500, 1, 1, 0, x_to_x_level=3, redundancy=False, func_seed=i + seed,
                                              x_seed=i + seed, linear=True)
        print(i)
        print(X_.columns)

        if toward == 'AB':
            output_DF = pd.DataFrame()
            output_DF[0] = x_
            output_DF[1] = y_exp_
            func_name = str(X_.columns[0]).replace('*', '×')
            output_DF.to_csv(rf'dataset\Linears\A→B\{func_name}_{i + seed}.csv', index=False, header=False)

        elif toward == 'BA':
            output_DF = pd.DataFrame()
            output_DF[0] = y_exp_
            output_DF[1] = x_
            func_name = str(X_.columns[0]).replace('*', '×')
            output_DF.to_csv(rf'dataset\Linears\B→A\{func_name}_{i + seed}.csv', index=False, header=False)


#
def gen_nonlinear(toward='AB', seed=0, num=100):
    for i in range(num):

        x_, X_, y_exp_, x_picked_ = gen_y_exp(500, 1, 1, 0, x_to_x_level=4, redundancy=True, func_seed=i + seed,
                                              x_seed=i + seed, linear=False, redun_ratio=0.5)
        print(i)
        print(X_.columns)

        func_name = '+'.join(list(X_.columns))
        func_name = func_name.replace('*', '×')
        print(func_name)

        if toward == 'AB':
            output_DF = pd.DataFrame()
            output_DF[0] = x_[x_picked_]
            output_DF[1] = y_exp_
            output_DF.to_csv(rf'dataset\Non_linears\A→B\{func_name}_{i + seed}.csv', index=False, header=False)

        elif toward == 'BA':
            output_DF = pd.DataFrame()
            output_DF[0] = y_exp_
            output_DF[1] = x_[x_picked_]
            output_DF.to_csv(rf'dataset\Non_linears\B→A\{func_name}_{i + seed}.csv', index=False, header=False)


gen_nonlinear(toward='AB', seed=0)
