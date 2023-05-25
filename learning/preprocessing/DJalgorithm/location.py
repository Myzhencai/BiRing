# https://blog.csdn.net/cunyizhang/article/details/112462524?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-112462524-blog-122627674.235^v27^pc_relevant_default_base1&spm=1001.2101.3001.4242.1&utm_relevant_index=3
import geomas as gm
import numpy as np
import matplotlib.pyplot as plt

def DJlocation(sensor1,sensor2,sensor3):
    #计算三个sensor向量的点法公式得到3个平面
    #点法公式：https://zhuanlan.zhihu.com/p/355542093得到每个面的参数[a,b,c,d]
    face1param = np.array([sensor1[0],sensor1[1],sensor1[2],
                           -(sensor1[0]**2 +sensor1[1]**2+sensor1[2]**2)])
    face2param = np.array([sensor2[0],sensor2[1],sensor2[2],
                           -(sensor2[0]**2 +sensor2[1]**2+sensor2[2]**2)])
    face3param = np.array([sensor3[0],sensor3[1],sensor3[2],
                           -(sensor3[0]**2 +sensor3[1]**2+sensor3[2]**2)])
    #计算sensor1与sensor2相交线向量\sensor1与sensor3相交线向量\sensor2与sensor3相交线向量
    vector_face1_2 = gm.Vector().calVectorFrom2Planes(face1param, face2param)
    vector_face1_3 = gm.Vector().calVectorFrom2Planes(face1param, face3param)
    vector_face2_3 = gm.Vector().calVectorFrom2Planes(face2param, face3param)

    #相交线的公式方法，这里是垂直于交线向量的求解:https://blog.csdn.net/lkysyzxz/article/details/81235445
    face1normalto_vector_face1_2 = np.cross(vector_face1_2,sensor1)
    face1normalto_vector_face1_3 = np.cross(vector_face1_3,sensor1)
    face2normalto_vector_face2_3 = np.cross(vector_face2_3,sensor2)
    #各个面上的交线上的一点
    face1point1 = sensor1
    face1point2 = sensor1 +0.5*(face1normalto_vector_face1_2)
    line_face1_2_point =gm.Coordinate().calCoordinateFrom2PointsAndPlane(face1point1, face1point2, face2param)

    face1point1 = sensor1
    face1point3 = sensor1 +0.5*(face1normalto_vector_face1_3)
    line_face1_3_point =gm.Coordinate().calCoordinateFrom2PointsAndPlane(face1point1, face1point3, face3param)

    face2point2 = sensor2
    face2point3 = sensor2 +0.5*(face2normalto_vector_face2_3)
    line_face2_3_point =gm.Coordinate().calCoordinateFrom2PointsAndPlane(face2point2, face2point3, face3param)

    # 计算两条直线在空间上的交点（我们有三条直线，这三条线在空间上两两会有一个交点或者只有一个或者没有）
    # line1_2point = gm.Coordinate().calCoordinateFrom2LinesByLS(line_face1_2_point, vector_face1_2, line_face1_3_point, vector_face1_3)
    # line1_3point = gm.Coordinate().calCoordinateFrom2LinesByLS(line_face1_2_point, vector_face1_2, line_face2_3_point, vector_face2_3)
    # line2_3point = gm.Coordinate().calCoordinateFrom2LinesByLS(line_face1_3_point, vector_face1_3, line_face2_3_point, vector_face2_3)
    line1_2point = gm.Coordinate().calCoordinateFrom2Lines(line_face1_2_point, vector_face1_2, line_face1_3_point, vector_face1_3)
    line1_3point = gm.Coordinate().calCoordinateFrom2Lines(line_face1_2_point, vector_face1_2, line_face2_3_point, vector_face2_3)
    line2_3point = gm.Coordinate().calCoordinateFrom2Lines(line_face1_3_point, vector_face1_3, line_face2_3_point, vector_face2_3)

    # print(line1_2point)
    # print(line1_3point)
    # print(line2_3point)

    if line1_2point[0] == line1_3point[0] ==line2_3point[0]:
        return np.array(line1_2point).reshape(3,)
    else:
        return np.zeros((3,))



def load_real_magdata(path):
    file = path
    all_data = np.loadtxt(file)
    mag_data = all_data[:, 6:15]
    # mag_data = mag_data[:, [0, 2, 1]]
    pos = all_data[:, 0:3]
    rot = all_data[:, 3:6]
    return mag_data, pos, rot


if __name__ =="__main__":
    real_magdata_path = "/home/gaofei/Aura/Aruadata/test_3traindata.txt"
    mag_data, pos, rot = load_real_magdata(real_magdata_path)
    #
    # sensor1 =np.array([9,5,7])
    # sensor2 =np.array([30,6,7])
    # sensor3 =np.array([9,5,8])
    #
    DJpos = []
    sensor1 =mag_data[:,:3]
    sensor2 =mag_data[:,3:6]
    sensor3 =mag_data[:,6:9]
    for i in range(sensor3.shape[0]):
        DJpos.append(DJlocation(sensor1[i], sensor2[i], sensor3[i]))
    print("hello")
    DJpos = np.array(DJpos).reshape(-1,3)*0.001

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num = 1000

    # print(tracking_pre[:100])
    # ax.scatter(tracking_raw[:num, 0], tracking_raw[:num, 1], tracking_raw[:num, 2], alpha=.5, marker='.')
    # ax.scatter(tracking_pre[:num, 0], tracking_pre[:num, 1], tracking_pre[:num, 2], alpha=.5, marker='.', c="r")
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], marker='.',c = "b")
    ax.scatter(DJpos[:,0], DJpos[:,1], DJpos[:,2], marker='.',c="r")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # print(tracking_raw)
    plt.show()


