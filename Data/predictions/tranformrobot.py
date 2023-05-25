#此处的代码为了将获得的机械臂数据转化到发射器的坐标系下（发射器绑定在机械臂末端）
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
from mpl_toolkits.mplot3d import Axes3D
#定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')

# 解析原始数据
datalistnew =[]
# with open('/home/gaofei/Aura/testnn.txt') as f:
with open('/home/gaofei/Aura/Aruadata/test_6.txt') as f:
    for line in f:
        listcurrentline = line.split("[")
        # print(line)
        # listcurrentline = line.split(" ")
        listcurrentline = listcurrentline[1].split("]")
        listcurrentline = listcurrentline[0].split(",")
        datalistnew.append(listcurrentline)

new_datasetnew=[]

for i in range(len(datalistnew)):
    for num in datalistnew[i]:
        new_datasetnew.append(float(num))
# print(new_datasetnew)

def eulerAnglesToRotationMatrix(theta):
    # 分别构建三个轴对应的旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    # 将三个矩阵相乘，得到最终的旋转矩阵
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

rawdatanew = np.array(new_datasetnew).reshape(-1,15)
# print(rawdatanew)
# np.savetxt("/home/gaofei/Aura/Aruadata/test_6traindata.txt",rawdatanew)

# 画出原始数据（在机械臂坐标系下）
# zdnew = rawdatanew[:,0]
# xdnew = rawdatanew[:,1]
# ydnew = rawdatanew[:,2]
#
# ax1.scatter3D(xdnew,ydnew,zdnew,c='red')  #绘制散点图
# plt.show()

# 固定的接收器的6dof值
reciver = np.array([679.3,-40.7,525,89.5,18.2,86.3])
reciverlocation = reciver[:3]
reciverrotationmatrix = R.from_euler('xyz', [89.5,18.2,86.3], degrees=True).as_matrix()

#sender 旋转平移表示(以sender为标准)
senderlocation = rawdatanew[:,:3]
sendereuler =  rawdatanew[:,3:6]
# print("rotation:",senderrotationamatrix)
transforvalue = reciverlocation - senderlocation
rotationlist =[]
print(senderlocation.shape[0])
for i in range(senderlocation.shape[0]):
    # print(sendereuler[i][0], sendereuler[i][1], sendereuler[i][2])
    currentsenderrotationmatrix_inv = R.from_euler('xyz', [sendereuler[i][0], sendereuler[i][1], sendereuler[i][2]], degrees=True).inv().as_matrix()
    reciver_to_senderrotaionmatrix = reciverrotationmatrix*currentsenderrotationmatrix_inv
#    旋转矩阵转欧拉角度
    reciver_to_senderrotaioneuler = R.from_matrix(reciver_to_senderrotaionmatrix).as_euler('xyz', degrees=True)
    # print(reciver_to_senderrotaioneuler)
    rotationlist.append(reciver_to_senderrotaioneuler)

rotationarray = np.array(rotationlist).reshape(-1,3)
print(rotationarray)

reciver_to_sender_6dof = np.hstack((transforvalue,rotationarray))
reciver_to_sender_6dofwithmag = np.hstack((reciver_to_sender_6dof,rawdatanew[:,6:]))
np.savetxt("/home/gaofei/Aura/Aruadata/reciver_to_sender_6dof.txt",reciver_to_sender_6dof)
np.savetxt("/home/gaofei/Aura/Aruadata/reciver_to_sender_6dofwithmag.txt",reciver_to_sender_6dofwithmag)

# 画出reciver 相对与sender的轨迹
zdnew = reciver_to_sender_6dof[:,0]
xdnew = reciver_to_sender_6dof[:,1]
ydnew = reciver_to_sender_6dof[:,2]

zd = rawdatanew[:,0]
xd = rawdatanew[:,1]
yd = rawdatanew[:,2]

rotx = reciver_to_sender_6dof[:,3]
roty = reciver_to_sender_6dof[:,4]
rotz = reciver_to_sender_6dof[:,5]

ax1.scatter3D(xdnew,ydnew,zdnew,c='red')  #reciver相对与sender
ax1.scatter3D(xd,yd,zd,c='b')  #reciver相对与sender
ax1.scatter3D(rotx,roty,rotz,c='y')  #reciver相对与sender


plt.show()
print(reciver_to_sender_6dof)


currentsenderrotationmatrix_inv = R.from_euler('xyz', [90, 0,0 ], degrees=True).inv().as_matrix()
reciverrotationmatrix =R.from_euler('xyz', [-90, 0,0 ], degrees=True).inv().as_matrix()
reciver_to_senderrotaionmatrix = reciverrotationmatrix*currentsenderrotationmatrix_inv
# 旋转矩阵转欧拉角度
reciver_to_senderrotaioneuler = R.from_matrix(reciver_to_senderrotaionmatrix).as_euler('xyz', degrees=True)
print("rotate angle ",reciver_to_senderrotaioneuler)









