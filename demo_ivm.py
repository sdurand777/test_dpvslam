import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.stream import image_stream, video_stream, image_ivm_300_stream
from dpvo.utils import Timer


from plyfile import PlyElement, PlyData

import open3d as o3d

import time

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):

    slam = None
    queue = Queue(maxsize=8)

    if os.path.isdir(imagedir):
        reader = Process(target=image_ivm_300_stream, args=(queue, imagedir, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    imagedir_img = imagedir+"/left/"
    number_of_images = len([file for file in os.listdir(imagedir_img) if file.endswith('.JPG')])
        
    print("number_of_images : ", number_of_images)

    counter = 0

    while 1:
       # (t, image, intrinsics) = queue.get()
        (t, image, disp_sens, intrinsics) = queue.get(timeout=10)
        if t < 0: break

        if t >= number_of_images-10:
           break

        counter += 1

        print("counter : ", counter)

        # image = torch.from_numpy(image).permute(2,0,1).cuda()
        # intrinsics = torch.from_numpy(intrinsics).cuda()

        image = image.to(torch.uint8).cuda()

        intrinsics = intrinsics.cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)


        image = image.cuda()

        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=timeit):
            slam(t, image, disp_sens, intrinsics)

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]
    points = np.array([(x,y,z,r,g,b) for (x,y,z),(r,g,b) in zip(points, colors)],
                      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(points, 'vertex',{'some_property': 'f8'},{'some_property': 'u4'})

    # Filtrer les points qui sont dans la distance maximale
    max_distance = 3.0

    # Calculer la distance euclidienne des points par rapport à l'origine
    x = points['x']
    y = points['y']
    z = points['z']
    
    distances = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    # Filtrer les points qui sont dans la distance maximale
    filtered_points = points[distances <= max_distance]

    el = PlyElement.describe(filtered_points, 'vertex', 
                             {'x': 'f4', 'y': 'f4', 'z': 'f4', 'red': 'u1', 'green': 'u1', 'blue': 'u1'})

    ply_data = PlyData([el], text=True)
    ply_data.write("output_test.ply")

# Lire le fichier PLY
    ply_data = PlyData.read("output_test.ply")

# Compter le nombre de points
    num_points = len(ply_data['vertex'].data)
    print(f"Nombre de points dans le fichier output PLY : {num_points}")

# Charger le fichier PLY
    ply_file_path = "output_test.ply"
#     pcd = o3d.io.read_point_cloud(ply_file_path)
#
# # Visualiser le point cloud
#     o3d.visualization.draw_geometries([pcd])

    print("ply saved ...")
    time.sleep(10)






    reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str, default='/home/ivm/20240823-12h10m44s-Pointe_Rouge-Escargot2_10x10/resized/left')
    parser.add_argument('--calib', type=str)
    parser.add_argument('--name', type=str, help='name your run', default='result')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_ply', action="store_true")
    parser.add_argument('--save_colmap', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    (poses, tstamps), (points, colors, calib) = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit)

    # trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)
    #
    # if args.save_ply:
    #     save_ply(args.name, points, colors)
    #
    # if args.save_colmap:
    #     save_output_for_COLMAP(args.name, trajectory, points, colors, *calib)
    #
    # if args.save_trajectory:
    #     Path("saved_trajectories").mkdir(exist_ok=True)
    #     file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)
    #
    # if args.plot:
    #     Path("trajectory_plots").mkdir(exist_ok=True)
    #     plot_trajectory(trajectory, title=f"DPVO Trajectory Prediction for {args.name}", filename=f"trajectory_plots/{args.name}.pdf")


        

