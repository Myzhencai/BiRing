import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
import pickle
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from mag_utils import DeviceCalibration, compute_sensor, get_field, get_field_new,compute_sensor_new,plot_correlation, get_sensor_pos
import matplotlib.pyplot as plt


# 加载我们用机械臂拿到的数据
def load_resampled_data_new(path):
    all_data = np.loadtxt(path)
    mag_data = all_data[:, 6:9]
    # mag_data = mag_data[:, [0, 2, 1]]#可能不需要调整
    pos = all_data[:, 0:3]*0.0001
    rot = all_data[:, 3:6]
    return mag_data, pos, rot


# 磁偶极子根据空间位置的预期读书
def dipole_model_newly(pos):
    # https://ccmc.gsfc.nasa.gov/RoR_WWW/presentations/Dipole.pdf
    pos = np.atleast_2d(pos)
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # print("r",r)
    Bx = 3*x*z/(r**5)
    By = 3*y*z/(r**5)
    Bz = (3*z**2-r**2)/(r**5)
    field = np.vstack((Bx, By, Bz)).T
    return field * 1e5



def ballpoints(Radius):
    center = [0, 0, 0]
    radius = Radius
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    pointX = x.reshape((10000,))
    pointY = y.reshape((10000,))
    pointZ = z.reshape((10000,))

    posXY = np.c_[pointX,pointY]
    posXYZ = np.c_[posXY,pointZ]

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # ax.scatter3D(posXYZ[:,0], posXYZ[:,1], posXYZ[:,2], color="red")
    #
    # # show
    # plt.show()

    return posXYZ



def main():
    # 换成球形的数据
    pos= ballpoints(20)
    dipolevalue_noratation = dipole_model_newly(pos)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    calibratex =dipolevalue_noratation[:,0]
    calibratey =dipolevalue_noratation[:,1]
    calibratez =dipolevalue_noratation[:,2]
    ax.scatter3D(calibratex, calibratey, calibratez, color="blue")
    ax.scatter3D(pos[:,0], pos[:,1], pos[:,2], color="red")

    plt.title("3D scatter plot")
    plt.show()

if __name__ == "__main__":
    main()
