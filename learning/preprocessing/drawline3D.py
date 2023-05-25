import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建画布
fig = plt.figure(figsize=(12, 8),facecolor='lightyellow')
# 创建 3D 坐标系
ax = fig.gca(fc='whitesmoke',projection ='3d')
# 绘制 3D 图形
ax.plot3D(xs=[3, 3.82615375],  # x 轴坐标
          ys=[3, 2.72612338],  # y 轴坐标
          zs=[3, 3.49240388],  # z 轴坐标
          zdir='z', c ='k',    # color
marker ='o',    # 标记点符号
mfc ='r',    # marker facecolor
mec ='g',    # marker edgecolor
ms = 10,  # size
)

ax.plot3D(xs=[3, 2.45248281],  # x 轴坐标
          ys=[3, 2.40345795],  # y 轴坐标
          zs=[3, 3.58682409],  # z 轴坐标
          zdir='z', c ='k',    # color
marker ='o',    # 标记点符号
mfc ='r',    # marker facecolor
mec ='g',    # marker edgecolor
ms = 10,  # size
)

ax.plot3D(xs=[3, 3.13302222],  # x 轴坐标
          ys=[3, 2.24559349],  # y 轴坐标
          zs=[3, 2.35721239],  # z 轴坐标
          zdir='z', c ='k',    # color
marker ='o',    # 标记点符号
mfc ='r',    # marker facecolor
mec ='g',    # marker edgecolor
ms = 10,  # size
)

# ax.plot(xs=[2, 0, 0, 0, 1, 2, 0, 1, 2],
#         ys=[0, 0, 2, 0, 1, 2, 0, 1, 2],
#         zs=[2, 0, 2, 0, 1, 2, 0, 0, 0],
#         ls=':',
# color ='grey',
# marker ='o',
# mfc ='r',
# mec ='g'
# )
# 设置坐标轴标题和刻度
ax.set(xlabel='X',
ylabel ='Y',
zlabel ='Z',
xticks = np.arange(0, 4, 0.5),
yticks = np.arange(0, 4, 0.5),
zticks = np.arange(0, 4, 0.5)
)
# 调整视角
ax.view_init(elev=20,  # 仰角
             azim=40  # 方位角
             )

# 显示图形
plt.show()