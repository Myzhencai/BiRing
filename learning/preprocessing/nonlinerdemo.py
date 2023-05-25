# coding=utf-8
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import numpy as np


# 目标函数
def fun(a,b,c,d):
    def v(x):
        return np.log2(1+x[0]*a/b)+np.log2(1+x[1]*c/d)
    return v
#限制条件函数
def con(a,b,i):
    def v(x):
        return np.log2(1 + x[i] * a / b)-5
    return v



if __name__ == "__main__":
    # 定义常量值
    args = [2, 1, 3, 4]  # a,b,c,d
    args1 = [2, 5, 6, 4]
    # 设置初始猜测值
    x0 = np.asarray((0.5, 0.5))
    #设置限制条件
    '''Equality constraint means that the constraint function result is
     to be zero whereas inequality means that it is to be non-negative'''
    cons = ({'type': 'ineq', 'fun': con(args1[0],args1[1],0)},
            {'type': 'ineq', 'fun': con(args1[2],args1[3],1)},
            )

    res = minimize(fun(args[0],args[1],args[2],args[3]), x0, constraints=cons)
    print(res.fun)
    print(res.success)
    print(res.x)
