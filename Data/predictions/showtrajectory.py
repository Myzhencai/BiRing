import numpy as np

from utils import load_segmented_data, load_extracted_vicon_data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    tracking_raw = np.loadtxt("/home/gaofei/BiRing/Data/posedata/posedatareal.txt")
    tracking_pre = np.loadtxt("/home/gaofei/BiRing/Data/posedata/posedatapredicet.txt")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num =1000
    print(tracking_pre[:100])
    ax.scatter(tracking_raw[:num,0], tracking_raw[:num,1], tracking_raw[:num,2], alpha=.5, marker='.')
    ax.scatter(tracking_pre[:num,0], tracking_pre[:num,1], tracking_pre[:num,2], alpha=.5, marker='.',c="r")
    # ax.scatter(tracking_raw[:,0], tracking_raw[:,1], tracking_raw[:,2], marker='.')
    # ax.scatter(tracking_pre[:,0], tracking_pre[:,1], tracking_pre[:,2], marker='.',c="r")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # print(tracking_raw)
    plt.show()



if __name__ == "__main__":
    main()