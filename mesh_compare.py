import open3d as o3d
import glob
import os
import numpy as np
import matplotlib.pyplot as plt 
from plyfile import PlyData
import os
from multiprocessing import Pool
from tqdm import tqdm
def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex

def vis(pc):
    data = [[], [], []]
    for p in pc:
        data[0].append(p[0])
        data[1].append(p[1])
        data[2].append(p[2])
    return data

def plot(data, inx):
    
    ax = plt.subplot(1, 4, inx, projection='3d')
    ax.view_init(elev=30, azim=-60)
    plt.axis('off')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim((-1, 1))
    # ax.set_ylim((-1, 1))
    # ax.set_zlim((-1, 1))
    ax.set_title(names[inx-1]) 
    ax.scatter(data[0], data[1], data[2], c = 'r', marker = 'o', alpha=0.5, s=1) 
    
def compare(f_id):
    _, id = os.path.split(f_id)
    id = id.split('.')[0]
    pc2skh_f_path = os.path.join(pc2skh_dir, id + ".ply")
    deepcad_f_path = os.path.join(deepcad_dir, id + ".ply")
    secad_f_path = os.path.join(secad_dir, id + '.obj')
    extrude_net_path = f_id

    if os.path.exists(pc2skh_f_path) and os.path.exists(deepcad_f_path) and os.path.exists(secad_f_path) and os.path.exists(extrude_net_path):
        pc2skh_pc = read_ply(pc2skh_f_path)
        deepcad_pc = read_ply(deepcad_f_path)

        extrude_net_mesh = o3d.io.read_triangle_mesh(extrude_net_path)
        extrude_net_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(extrude_net_mesh, number_of_points=2000)
        extrude_net_pc = np.array(extrude_net_pc.points)

        secad_mesh = o3d.io.read_triangle_mesh(secad_f_path)
        secad_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(secad_mesh, number_of_points=2000)
        secad_pc = np.array(secad_pc.points)
        
        pc2skh_vis = vis(pc2skh_pc)
        deepcad_vis = vis(deepcad_pc)
        secad_vis = vis(secad_pc)
        extrude_vis = vis(extrude_net_pc)

        vis_lst = [pc2skh_vis, deepcad_vis, secad_vis, extrude_vis]
        
        fig=plt.figure(figsize=(10,10)) 
        for inx, p in enumerate(vis_lst):
            plot(p, inx=inx+1)

        fig_path = os.path.join(output_dir, id + '.png')
        plt.savefig(fig_path, dpi=500, bbox_inches='tight') 
        plt.close()
    return

if __name__ == '__main__':
    
    output_dir = "/data/wc/extrude_net/data"
    
    pc2skh_dir = "/data/wc/Points2sketch/proj_log/pc2skh-50-test/output/47/h5_rc_pc"
    deepcad_dir = "/data/wc/point2sequence/proj_log/pc2cad/pc2cad/results/fake_z_ckptlatest_num20000_dec_pc"
    secad_dir = "/data/wc/extrude_net/samples/pc2skh"
    extrude_net_dir= "/data/wc/extrude_net/samples/pc2skh"
    
    names =["pc2skh", "deepcad", "secad", "extrude"]

    file_lst = glob.glob(os.path.join(extrude_net_dir, "*.ply"))[:3]
    
    po = Pool(64)
    pbar = tqdm(file_lst, total=len(file_lst))
    for f_id in pbar:
        po.apply(compare, args=(f_id, ))
            
    po.close()
    po.join()