import os
import glob
import random
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

def readIndex(index_path, shuffle=False):
    f_lst = []
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            f_lst.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(f_lst)
    return f_lst

class ShapeNet(Dataset):
    '''
    Lazy Loading ShapeNet Dataset with PointCloud and Occupancy
    '''
    def __init__(self, data_path, implicite_function="Occupancy", balance=False, num_surface_points=2048, num_testing_points=2048):
        super().__init__()
        self.data_path = data_path
        self.balance = balance
        self.implicite_function = implicite_function
        self.num_surface_points = num_surface_points
        self.num_testing_points = num_testing_points
        self.data_urls = readIndex(data_path)

    def __getitem__(self, item):
        '''
        :param item: int
        :return: surface points [N, 3]
        :return: testing points with last bit indicating occupancy [M, 4]
        '''
        data_url = self.data_urls[item]

        surface_point_path = data_url[0]
        gt_path = data_url[1]
        # Loading Data file
        pointcloud = np.asarray(o3d.io.read_point_cloud(surface_point_path).points)

        # Load testing points
        if self.implicite_function == "Occupancy":
            testing_points = np.load(gt_path)
            # downsample testing point clouds
            if self.balance:
                inner_points = testing_points[testing_points[:,-1]==1]
                outer_points = testing_points[testing_points[:,-1]==0]
                inner_index = np.random.randint(0, inner_points.shape[0], self.num_testing_points//2)
                outer_index = np.random.randint(0, outer_points.shape[0], self.num_testing_points//2)
                testing_points = np.concatenate([inner_points[inner_index], outer_points[outer_index]], axis=0)
            else:
                testing_indices = np.random.randint(0, testing_points.shape[0], self.num_testing_points)
                testing_points = testing_points[testing_indices]
        else:
            testing_points = np.load(gt_path)
            # downsample testing point clouds
            if self.balance:
                inner_points = testing_points[testing_points[:,-1]<0]
                outer_points = testing_points[testing_points[:,-1]>=0]
                inner_index = np.random.randint(0, inner_points.shape[0], self.num_testing_points//2)
                outer_index = np.random.randint(0, outer_points.shape[0], self.num_testing_points//2)
                testing_points = np.concatenate([inner_points[inner_index], outer_points[outer_index]], axis=0)
            else:
                testing_indices = np.random.randint(0, testing_points.shape[0], self.num_testing_points)
                testing_points = testing_points[testing_indices]
        # downsample surface point clouds
        surface_indices = np.random.randint(0, pointcloud.shape[0], self.num_surface_points)
        pointcloud = pointcloud[surface_indices]
        return pointcloud.astype(np.float32), testing_points.astype(np.float32)

    def __len__(self):
        return len(self.data_urls)
    
    

if __name__ == "__main__":
    dataset = ShapeNet(data_path="/data/wc/extrude_net/data/pc2skh/extrudenet/data_train.txt", balance=False)
    print(dataset[0])
    for i in dataset:
        continue