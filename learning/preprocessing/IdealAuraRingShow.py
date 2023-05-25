import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
import pickle
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from mag_utils import DeviceCalibration, compute_sensor, get_field, get_field_new,compute_sensor_new,plot_correlation, get_sensor_pos


CROSSTALK_MATRIX = False
CROSS_BIAS = False
POLY_FIT = False
POLY_ORDER = 11
SENSOR_ROT = False

# 需要学习的参数
GLOBAL_GAIN =True
GAIN =True
PER_CHANNEL_GAIN =True
BIAS  =True
NOISE =True
SENSOR_OFFSET =True
SENSOR_ROT_OFFSET =True
RING_OFFSET =True
RING_ROT_OFFSET =True
CROSSTALK =True
BASE_SENSOR_OFFSET =True
BASE_SENSOR_ROT_OFFSET =True
USE_INTERP_CORRECTION = True
USE_SIGN_CHANGE = False




# 解码一维的需要训练的参数
def decode_x(data):
    i = 0
    global_gain = data[i]
    i += 1
    gain = data[i:i + 3]
    i += 3
    per_channel_gain = data[i:i + 3]
    i += 3
    bias = data[i:i + 3]
    i += 3
    noise = data[i:i + 3]
    i += 3
    sensor_offset = data[i:i + 3]
    i += 3
    # 此处切换为欧拉角
    sensor_rot_offset = data[i:i + 3]  # x, y, z,
    i += 3
    ring_offset = data[i:i + 3]
    i += 3
    ring_rot_offset = data[i:i + 3]
    i += 3
    crosstalk = data[i:i + 3]
    i += 3
    base_sensor_offset = data[i:i + 3]
    i += 3
    base_sensor_rot_offset = data[i:i + 3]
    i += 3
    return global_gain,gain,per_channel_gain,bias,noise,sensor_offset,sensor_rot_offset,\
        ring_offset,ring_rot_offset,crosstalk,base_sensor_offset,base_sensor_rot_offset


# 编码一维的需要训练的参数
def encode_x(global_gain,gain,per_channel_gain,bias,noise,sensor_offset,sensor_rot_offset,
                  ring_offset,ring_rot_offset,crosstalk,base_sensor_offset,base_sensor_rot_offset):
    x = []
    if GLOBAL_GAIN:
        x +=global_gain
    if  GAIN:
        x +=gain
    if PER_CHANNEL_GAIN:
        x += per_channel_gain
    if BIAS:
        x += bias
    if NOISE:
        x += noise
    if SENSOR_OFFSET:
        x += sensor_offset
    if SENSOR_ROT_OFFSET:
        x += sensor_rot_offset
    if RING_OFFSET:
        x += ring_offset
    if RING_ROT_OFFSET:
        x += ring_rot_offset
    if CROSSTALK:
        x += crosstalk
    if BASE_SENSOR_OFFSET:
        x +=base_sensor_offset
    if BASE_SENSOR_ROT_OFFSET:
        x +=base_sensor_rot_offset
    return np.array(x)








# 磁偶极子根据空间位置的预期读书
def dipole_model_newly(pos):
    # https://ccmc.gsfc.nasa.gov/RoR_WWW/presentations/Dipole.pdf
    pos = np.atleast_2d(pos)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    Bx = 3*x*z/(r**5)
    By = 3*y*z/(r**5)
    Bz = (3*z**2-r**2)/(r**5)
    field = np.vstack((Bx, By, Bz)).T
    return field * 1e5


def compute_sensor_newly(pos, rot, x):
    # 解析出当前的需要优化的参数
    global_gain, gain, per_channel_gain, bias, noise, sensor_offset, sensor_rot_offset, \
        ring_offset, ring_rot_offset, crosstalk, base_sensor_offset, base_sensor_rot_offset = decode_x(x)
    kxy,kxz,kyz =0.,0.,0.

    # 计算场强值（改成旋转矩阵形式）
    ring_q =rot
    #旋转矩阵运算和apply运算的结果需要转置
    sensor = base_sensor_offset + sensor_offset

    # 改成旋转矩阵形式
    # sensor_rot = sensor_rot_offset * base_sensor_rot_offset
    sensor_rot_offset = R.from_euler('xyz', sensor_rot_offset, degrees=True).as_matrix()
    base_sensor_rot_offset = R.from_euler('xyz', base_sensor_rot_offset, degrees=True).as_matrix()
    sensor_rot = np.matmul(sensor_rot_offset,base_sensor_rot_offset)

    # 改成欧拉角和旋转矩阵
    # ring_qs = R.from_quat(ring_q)
    ring_qs = R.from_euler('xyz', ring_q, degrees=True).as_matrix()
    # ring_pos_adj = pos + ring_qs.apply(ring_offset, inverse=True)
    ring_pos_adj = pos + np.matmul(np.linalg.inv(ring_qs) ,ring_offset)
    # ring_rot_adj = ring_qs * ring_rot_offset
    ring_rot_offset = R.from_euler('xyz', ring_rot_offset, degrees=True).as_matrix()
    ring_rot_adj = np.matmul(ring_qs,ring_rot_offset)

    # sensor_ring = (ring_qs * ring_rot_offset).apply(sensor - ring_pos_adj)
    vetornew = (sensor - ring_pos_adj).reshape(-1,3,1)
    sensor_ring = np.matmul(ring_rot_adj,vetornew).reshape(-1,3)
    # field = ring_rot_adj.apply(dipole_model_newly(sensor_ring), inverse=True)
    dipolevalue = dipole_model_newly(sensor_ring).reshape(-1,3,1)
    field = np.matmul(np.linalg.inv(ring_rot_adj) ,dipolevalue)
    # field = sensor_rot.apply(field)
    field = np.matmul(sensor_rot,field).reshape(-1,3)

    # 计算根据公式计算理论的预测读数值
    field_adj = field * global_gain / per_channel_gain
    # field_adj = field * global_gain
    # 此处的公式和Aruraring的差异较大一个用的field一个用的field_adj,同时引入了更多的协防差
    # field_adj = np.abs(field * global_gain / per_channel_gain)
    coeffs = np.array([[1, kxy ** 2, kxz ** 2],
                       [kxy ** 2, 1, kyz ** 2],
                       [kxz ** 2, kyz ** 2, 1],
                       [2 * kxy * kxz, 0, 0],
                       [0, 2 * kxy * kyz, 0],
                       [0, 0, 2 * kxz * kyz],
                       noise ** 2])

    features_1 = np.hstack((field_adj ** 2,
                            np.array([field_adj[:, 1] * field_adj[:, 2],
                                      field_adj[:, 0] * field_adj[:, 2],
                                      field_adj[:, 0] * field_adj[:, 1]]).T, np.ones((field_adj.shape[0], 1))))
    sensors = gain * np.sqrt(np.matmul(features_1, coeffs)) - bias
    # 对应的是公式
    sensors[sensors < 0] = 0

    return sensors



def ballpoints(Radius):
    center = [0, 0, 0]
    radius = Radius
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    pointX = x.reshape((40000,))
    pointY = y.reshape((40000,))
    pointZ = z.reshape((40000,))

    posXY = np.c_[pointX,pointY]
    posXYZ = np.c_[posXY,pointZ]

    return posXYZ

def main():
    # 加载实际的Sensor的读数以及对应的Pose和Rotation数据
    pos = ballpoints(20)
    rot = np.zeros((40000,3))
    rot1 = np.ones((40000,3))*20

    # 初始化需要优化的参数
    global_gain = [20]
    gain = [700,900,908]
    per_channel_gain = [500,500,500]
    bias = [0.1,0.1,0.1]
    noise =[2,2,2]
    sensor_offset = [-4,8,8]

    # global_gain = [0]
    # gain = [0,0,0]
    # per_channel_gain = [0,0,0]
    # bias = [0.,0.,0.]
    # noise =[0,0,0]
    # sensor_offset = [0,0,0]

    # 将所有的四元数变为欧拉角
    sensor_rot_offset =[120,60,20]
    ring_offset = [8,9,9]
    ring_rot_offset =[42,45,90]
    crosstalk = [3,2,4]
    base_sensor_offset = [0.2,0.4,0.5]
    base_sensor_rot_offset = [20,39,28]

    # sensor_rot_offset =[0,0,0]
    # ring_offset = [0,0,0]
    # ring_rot_offset =[0,0,0]
    # crosstalk = [0,0,0]
    # base_sensor_offset = [0,0,0]
    # base_sensor_rot_offset = [0,0,0]

    # 将需要优化的参数打包
    x0 = encode_x(global_gain,gain,per_channel_gain,bias,noise,sensor_offset,sensor_rot_offset,
                  ring_offset,ring_rot_offset,crosstalk,base_sensor_offset,base_sensor_rot_offset)


    calibratedmag = compute_sensor_newly(pos, rot, x0)
    calibratedmag1 = compute_sensor_newly(pos, rot1, x0)

    # 真实数据

    def merge4(filePath):
        area0 = np.loadtxt(filePath + "area0.txt")
        area1 = np.loadtxt(filePath + "area1.txt")
        area2 = np.loadtxt(filePath + "area2.txt")
        area3 = np.loadtxt(filePath + "area3.txt")

        mergeddata = np.r_[area0, area1]
        mergeddata = np.r_[mergeddata, area2]
        mergeddata = np.r_[mergeddata, area3]
        # print(mergeddata.shape)

        np.savetxt(filePath + "megedData.txt", mergeddata)

    leftdatafilepath = "/home/gaofei/PycharmProjectss/MAuto2-main/Data/leftdata/"
    # rightdatafilepath = "/home/gaofei/PycharmProjectss/MAuto2-main/Data/rightdata/"
    merge4(leftdatafilepath)
    # merge4(rightdatafilepath)
    datapathleft = leftdatafilepath + "megedData.txt"
    # datapathright = rightdatafilepath + "megedData.txt"
    rawdata = np.loadtxt(datapathleft)[:, :3]
    newrawdata = []
    for i in range(rawdata.shape[0]):
        if rawdata[i][0]>2000 or rawdata[i][1]>2000 or rawdata[i][2]>2000:
            print("剔除的",rawdata[i])
        else:
            newrawdata.append(rawdata[i])
    newrawdata = np.array(newrawdata).reshape((-1,3))

    # 画图
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    calibratex =calibratedmag[:,0]
    calibratey =calibratedmag[:,1]
    calibratez =calibratedmag[:,2]
    ax.scatter3D(calibratex, calibratey, calibratez, color="red")

    calibratex1 =calibratedmag1[:,0]
    calibratey1 =calibratedmag1[:,1]
    calibratez1 =calibratedmag1[:,2]
    # ax.scatter3D(calibratex1, calibratey1, calibratez1, color="green")

    # x = rawdata[:, 0]
    # y = rawdata[:, 1]
    # z = rawdata[:, 2]
    # ax.scatter3D(x, y, z, color="blue")

    x = newrawdata[:, 0]
    y = newrawdata[:, 1]
    z = newrawdata[:, 2]
    ax.scatter3D(x, y, z, color="blue")
    plt.title("3D caculated plot")
    plt.show()

    def load_resampled_data_new(path):
        all_data = np.loadtxt(path)
        mag_data = all_data[:, 6:9]
        # mag_data = mag_data[:, [0, 2, 1]]#可能不需要调整
        pos = all_data[:, 0:3] * 0.0001
        rot = all_data[:, 3:6]
        return mag_data, pos, rot

#     测试真实数据与理论数据
#     dataPath = "/home/gaofei/Aura/Aruadata/reciver_to_sender_6dofwithmag.txt"
#     mag_datareal, posreal, rotreal = load_resampled_data_new(dataPath)
#     mag_dataideal = compute_sensor_newly(posreal, rotreal, x0)
    # 理想
    # x = mag_dataideal[:, 0]
    # y = mag_dataideal[:, 1]
    # z = mag_dataideal[:, 2]
    # ax.scatter3D(x, y, z, color="red")
    # 现实状况

    # newrawdatareal = []
    # for i in range(mag_datareal.shape[0]):
    #     if mag_datareal[i][0] > 2000 or mag_datareal[i][1] > 2000 or mag_datareal[i][2] > 2000:
    #         print("剔除的", mag_datareal[i])
    #     else:
    #         newrawdatareal.append(mag_datareal[i])
    # newrawdatareal = np.array(newrawdatareal).reshape((-1, 3))
    # xr = newrawdatareal[:, 0]
    # yr = newrawdatareal[:, 1]
    # zr = newrawdatareal[:, 2]
    # ax.scatter3D(xr, yr, zr, color="blue")

    # plt.show()



#1.改动替换四元数运算为矩阵运算即可

if __name__ == "__main__":
    main()
