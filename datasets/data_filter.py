import sys
sys.path.append('..')
sys.path.append('.')
from dataset import readIndex
import os
import numpy as np
import math
import h5py
from tqdm import tqdm
from multiprocessing import Pool
import open3d as o3d

def process(data_url):
    surface_point_path = data_url[0]
    pointcloud = np.asarray(o3d.io.read_point_cloud(surface_point_path).points)
    if pointcloud.shape[0] == 0:
        return False
    else:
        return True

if __name__ == "__main__":
    # data_source = "/data/wc/extrude_net/data/pc2skh/extrudenet/train.txt"
    # new_txt = "/data/wc/extrude_net/data/pc2skh/extrudenet/train_new.txt"
    # data_urls = readIndex(data_source)
    # pbar = tqdm(data_urls, total=len(data_urls))
    # data_urls_new = []
    # for d in pbar:
    #     if process(d):
    #         data_urls_new.append(d)
    #     else:
    #         continue
    # with open(new_txt, 'w') as f:
    #     for i,j in data_urls_new:
    #         f.write(i + ' ' + j + '\n')
    
    pointcloud = np.asarray(o3d.io.read_point_cloud("/data/wc/Points2sketch/DATABASE/deepcad/pc2skh_3d/mesh/0093/00933512_1_surface_point_cloud.ply").points)
    print(pointcloud.shape)
    
    
        


