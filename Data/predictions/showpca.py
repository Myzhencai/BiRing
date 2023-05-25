import numpy as np
import matplotlib.pyplot as plt

def pca(x, n):
    """
    :param x:输入数据,ndarray p*n维，每个数据具有p维特征，共n个数据组成数据集x
    :param n:取得前n个特征向量
    """
    # step1：数据集标准化
    sum_x = np.sum(x, axis=1)
    mean_x = (sum_x / x.shape[1]).reshape(x.shape[0], 1)
    MEAN_X = np.tile(mean_x, reps=x.shape[1])
    x_ = x - MEAN_X  # 得到去中心化的数据x_
    # step2:求x_*T(x)的协方差矩阵
    T_x = x_.T
    Cov = (1 / (x.shape[1] - 1)) * np.matmul(x_, T_x)  # 得到协方差矩阵
    # step3:对协方差矩阵进行特征分解
    eigenvalue, featurevector = np.linalg.eig(Cov)
    # step4:特征向量按照特征值大小进行排序
    for i in range(len(eigenvalue) - 1):
        for j in range(len(eigenvalue) - i - 1):
            if abs(eigenvalue[j]) < abs(eigenvalue[j + 1]):
                eigenvalue[j], eigenvalue[j + 1] = eigenvalue[j + 1], eigenvalue[j]
                featurevector[:, [j, j + 1]] = featurevector[:, [j + 1, j]]
    # step5:得到变换矩阵
    U = featurevector[:, 0:n]
    # step6:pca降维
    y = np.matmul(U, x)

    return y

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    x0 = np.array([[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
                  [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]])
    U, pca_x = pca(x0, 2)

    cxy = np.array([[1, 0], [0, 1]])  # 原空间基底
    new_cxy = np.matmul(U, cxy)
    x = np.linspace(-5, 5, 100)
    y1 = (new_cxy[1, 0] / new_cxy[0, 0]) * x
    y2 = (new_cxy[1, 1] / new_cxy[0, 1]) * x
    plt.figure(1, figsize=(8, 20))
    ax1 = plt.subplot(2, 1, 1)  # （行，列，活跃区）
    plot_point(x0,"原始数据点")
    plt.plot(x, y1, color='r',linewidth=2, label="新基底(第一主成分)")
    plt.plot(x, y2, color='g', linewidth=2,label="新基底(第二主成分)")
    plt.plot(x, np.full(fill_value=0, shape=x.shape), 'k--', linewidth=2, label="原基底")
    plt.plot(np.full(fill_value=0, shape=x.shape), x, 'k--', linewidth=2)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.grid(True)
    plt.legend(loc='best')

    ax2 = plt.subplot(2,2,3)
    first = np.matmul(U[:,0],x0)
    plt.scatter(first,np.full(fill_value=0, shape=first.shape),label="第一主成分降维")
    plt.grid(True)
    plt.legend(loc='best')

    ax3 = plt.subplot(2, 2, 4)
    second = np.matmul(U[:,1],x0)
    plt.scatter(second, np.full(fill_value=0, shape=second.shape), label="第二主成分降维")
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
