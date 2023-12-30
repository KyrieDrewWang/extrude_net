import os
import glob
import random
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from plyfile import PlyData, PlyElement


def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

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

    def get_by_index(self, id):
        
        base = "/data/wc/Points2sketch/DATABASE/deepcad/pc2skh_3d/mesh/"
        
        
        surface_point_path = os.path.join(base, id[:4], id+"_surface_point_cloud.ply")
        gt_path = os.path.join(base, id[:4], id+"_occupancy.npy")
        
        # Loading Data file
        pointcloud = np.asarray(o3d.io.read_point_cloud(surface_point_path).points)
        
        pointcloud = pointcloud[[not np.all(pointcloud[i] == 0) for i in range(pointcloud.shape[0])], :]
        
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

    def __getitem__(self, item):
        '''
        :param item: int
        :return: surface points [N, 3]
        :return: testing points with last bit indicating occupancy [M, 4]
        '''
        data_url = self.data_urls[item]

        surface_point_path = data_url[0]
        gt_path = data_url[1]
        _, data_id = os.path.split(data_url[0])  
        data_id = data_id.replace("_surface_point_cloud.ply", "")
        # Loading Data file
        pointcloud = np.asarray(o3d.io.read_point_cloud(surface_point_path).points)
        
        pointcloud = pointcloud[[not np.all(pointcloud[i] == 0) for i in range(pointcloud.shape[0])], :]
        
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
        return pointcloud.astype(np.float32), testing_points.astype(np.float32), data_id

    def __len__(self):
        return len(self.data_urls)
    
    

if __name__ == "__main__":
    dataset = ShapeNet(data_path="/data/wc/extrude_net/data/pc2skh/train.txt", balance=False)
    pc ,gt, id = dataset[0]
    write_ply(pc, "pc.ply")
    write_ply(gt[gt[:,3]==1], "gt_pc.ply")