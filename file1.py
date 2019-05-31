import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions


def sig_func(x):
    return 1/(1 + np.exp(-x))


def cal_unj(w1, x_ni, i, j, ip_c):
    sum = 0
    for k in range(ip_c + 1):  # 5
        sum = sum + x_ni[i][k] * w1[j][k]  # unj
    return sum


def cal_u_nk(w2, vnj, ip_c, j2):
    sum = 0;
    for k in range(ip_c + 2):       # 6
        sum = sum + vnj[k]*w2[j2][k]
    return sum


def main():
    ds = pd.read_csv('iris.csv')
    x = ds.iloc[1:120, 1:5]     # 120*4
    y = ds.iloc[1:120, 6:9]     # 120*3
    x_t = ds.iloc[121:150, 1:5]
    y_t = ds.iloc[121:150, 6:9]
    x_test = np.array(x_t)
    y_test = np.array(y_t)
    test_r, test_c = x_t.shape
    op_t_r, op_t_c = y_t.shape
    xx = np.array(x)
    y_nk = np.array(y)
    ip_r, ip_c = x.shape
    op_r, op_c = y.shape
    x_ni = np.c_[np.ones(ip_r), xx]     # adding bias column
    x_test_ni = np.c_[np.ones(test_r), x_test]
    lr = 0.16
    wji = np.random.random(size=(ip_c + 1, ip_c + 1))   # 5*5
    w_kj = np.random.random(size=(op_c, ip_c + 2))    # 3*6
    epsilon = 0.1
    E = 1
    cnt = 0
    iter = []
    E_list = []
    while E > epsilon:
        E = 0
        for i in range(ip_r):
            unj = []
            u_nk = []
            v_nk = []
            vnj = []
            bias = 1.0
            e = 0
            vnj.append(bias)
            s2 = 0
            for j in range(ip_c + 1):   # 5
                unj.append(cal_unj(wji, x_ni, i, j, ip_c))
                vnj.append(sig_func(unj[j]))
            for j2 in range(op_c):      # 3
                u_nk.append(cal_u_nk(w_kj, vnj, ip_c, j2))
                v_nk.append(sig_func(u_nk[j2]))
                e = e + math.pow(y_nk[i][j2] - v_nk[j2], 2)*1/2
                p = (y_nk[i][j2] - v_nk[j2])*sig_func(u_nk[j2])*(1 - sig_func(u_nk[j2]))
                for jj2 in range(ip_c + 2):     # 6
                    w_kj[j2][jj2] = w_kj[j2][jj2] + lr*p*vnj[jj2]
            for j4 in range(ip_c + 1):     # 5
                for k3 in range(op_c):      # 3
                    s2 = s2 + (y_nk[i][k3] - v_nk[k3])*sig_func(u_nk[k3])*(1 - sig_func(u_nk[k3]))*w_kj[k3][j4 + 1]*sig_func(unj[j4])*(1 - sig_func(unj[j4]))
                for i4 in range(ip_c + 1):  # 5
                    wji[j4][i4] = wji[j4][i4] + lr*s2*x_ni[i][i4]
            E = E + e
        cnt = cnt + 1
        iter.append(cnt)
        E = E/ip_r
        E_list.append(E)
        print(cnt, E)
    # plot
    plt.scatter(iter, E_list)
    plt.title("Error Graph")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()
    print(w_kj)
    print(end="\n")
    print(wji)
    print(end="\n")
    for i in range(test_r):
        ut_nj = []
        vt_nj = []
        bias2 = 1
        vt_nj.append(bias2)
        u_t_nk = []
        v_t_nk = []
        c_t = []
        for j in range(ip_c + 1):
            ut_nj.append(cal_unj(wji, x_test_ni, i, j, ip_c))
            vt_nj.append(sig_func(ut_nj[j]))
        for j2 in range(op_c):
            u_t_nk.append(cal_u_nk(w_kj, vt_nj, ip_c, j2))
            v_t_nk.append(sig_func(u_t_nk[j2]))
            c_t.append(v_t_nk)
        for j3 in range(op_c):
            print(y_test[i][j3], end="    ")
        print(end="      ")
        for j4 in range(op_c):
            print(v_t_nk[j4], end="     ")
        print(end="\n")


main()



