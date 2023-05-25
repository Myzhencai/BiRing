#此代码使用scipy来非线性优化AuraRing中的一些参数
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
import pickle
import seaborn as sns
from scipy.spatial.transform import Rotation as R

# 磁偶极子的计算公式，1e5为真空的磁场导值
def dipole_model(pos):
    # https://ccmc.gsfc.nasa.gov/RoR_WWW/presentations/Dipole.pdf
    # slide 2
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

# 根据提供的空间位置和旋转关系计算理论上的磁偶极子模型的
def get_field(ring_pos, ring_q, calib):
    # 直接通过空间位置和旋转来得到理论的场强值
    ring_q = np.atleast_2d(ring_q)[:, [1,2,3,0]]
    sensor = calib.base_sensor_offset + calib.sensor_offset
    sensor_rot = calib.sensor_rot_offset * calib.base_sensor_rot_offset

    ring_qs = R.from_quat(ring_q)
    ring_pos_adj = ring_pos + ring_qs.apply(calib.ring_offset, inverse=True)
    ring_rot_adj = ring_qs * calib.ring_rot_offset

    sensor_ring = (ring_qs * calib.ring_rot_offset).apply(sensor-ring_pos_adj)
    field = ring_rot_adj.apply(dipole_model(sensor_ring), inverse=True)

    return sensor_rot.apply(field)

def compute_sensor(field, calib):
    # 对应的还没找到先选择理解为auraring中的读数
    field_adj = field * calib.global_gain / calib.per_channel_gain
    coeffs = np.array([  [1, calib.kxy ** 2, calib.kxz ** 2],
                [calib.kxy ** 2, 1, calib.kyz ** 2],
                [calib.kxz ** 2, calib.kyz ** 2, 1],
                [2 * calib.kxy * calib.kxz, 0, 0],
                [0, 2 * calib.kxy * calib.kyz, 0],
                [0, 0, 2 * calib.kxz * calib.kyz],
                calib.noise ** 2])

    features_1 = np.hstack((field_adj ** 2,
    np.array([field_adj[:, 1]*field_adj[:, 2],
    field_adj[:, 0]*field_adj[:, 2],
    field_adj[:, 0]*field_adj[:, 1]]).T, np.ones((field_adj.shape[0], 1))))
    sensors = calib.gain * np.sqrt(np.matmul(features_1, coeffs)) - calib.bias

    sensors[sensors < 0] = 0
    return sensors

# 打包所有的参数放入cost函数中进行优化
def encode_x(per_channel_gain, crosstalk, cross_bias, bias, poly):
    x = []
    if PER_CHANNEL_GAIN:
        x += per_channel_gain
    if CROSSTALK_MATRIX:
        x += list(crosstalk.flatten())
    if CROSS_BIAS:
        x += cross_bias
    if BIAS:
        x += bias
    if POLY_FIT:
        x += poly
    return np.array(x)


def decode_x(x):
    i = 0

    if PER_CHANNEL_GAIN:
        per_channel_gain = x[i:i+3]
        i += 3
    else:
        per_channel_gain = [1, 1, 1]

    if CROSSTALK_MATRIX:
        crosstalk = x[i:i+9].reshape(3, 3)
        i += 9
    else:
        crosstalk = np.eye(3)

    if CROSS_BIAS:
        cross_bias = x[i:i+3]
        i += 3
    else:
        cross_bias = [0, 0, 0]

    if BIAS:
        bias = x[i:i+3]
    else:
        bias = [0, 0, 0]

    if POLY_FIT:
        poly = x[i:i+POLY_ORDER]
    else:
        poly = [0] * (POLY_ORDER-2) + [1, 0]
    return per_channel_gain, crosstalk, cross_bias, bias, poly




def auraRingparameters():
    print("training for aruraRing pameters")
    x0 = np.hstack((np.eye(3).flatten(), [0,0,0]))
    per_channel_gain = [1,1,1]
    crosstalk = np.eye(3)
    cross_bias = [0, 0, 0]
    bias = [0, 0, 0]
    poly = [0] * (POLY_ORDER-2) + [1, 0]
    x0 = encode_x(per_channel_gain, crosstalk, cross_bias, bias, poly)