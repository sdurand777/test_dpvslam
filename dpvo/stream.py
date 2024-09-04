import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain
import glob

import torch
from torch.nn import functional as F

def image_stream(queue, imagedir, calib, stride, skip=0):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))




def image_ivm_100_stream(queue, imagedir, stride, skip=0):
    """ image generator """

    img_exts = ["*.png", "*.jpeg", "*.jpg", "*.JPG"]

    print("imagedir : ", imagedir)

    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    print("image_list : ", image_list)

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))

       
        print("imfile : ", imfile)

        h, w, _ = image.shape

        img = image
        H, W, _ = img.shape
        
        # rectification
        K_r = np.array([4.6011859130859375e+02, 0., 3.0329241943359375e+02, 0., 4.6011859130859375e+02, 2.4381889343261719e+02, 0., 0., 1. ]).reshape(3,3)
        d_r = np.array([-3.34759861e-01, 1.55759037e-01, 7.29110325e-04,
                        1.10754154e-04, -4.32639048e-02 
                        ]).reshape(5)
        R_r = np.array([9.9999679129872809e-01, -2.5332565953597990e-03,
                        1.8083868698132711e-06, 2.5332551721412530e-03,
                        9.9999663218813351e-01, 5.6411933458219384e-04,
                        -3.2374398044074352e-06, -5.6411294338637344e-04,
                        9.9999984088304039e-01 
                        ]).reshape(3,3)

        P_r = np.array([ 4.5990068054199219e+02, 0., 3.3639562416076660e+02,
                        5.9697221843133875e+01, 0., 4.5990068054199219e+02,
                        2.6883334159851074e+02, 0., 0., 0., 1., 0. 
                        ]).reshape(3,4)

        map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (616, 514), cv2.CV_32F)

        intrinsics_vec = [4.5990068054199219e+02, 4.5990068054199219e+02, 3.3639562416076660e+02, 2.6883334159851074e+02]
        ht0, wd0 = [376, 514]

        images = [cv2.remap(img, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]

        images = torch.from_numpy(np.stack(images, 0))

        image_tmp = images.numpy().squeeze(0)
        # cv2.imshow("title", image_tmp)
        # cv2.waitKey(0)
 
        #images = images.permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
        images = images.permute(0, 3, 1, 2)

        # Ensure either size or scale_factor is defined

        #image_size = [448, 736]
        image_size = [528, 960]
        #image_size = [514, 616]

        #print("++++++++ image_size  : ",image_size)
        if image_size is not None:
            images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
        else:
            raise ValueError("image_size must be defined")
            
        intrinsics = torch.as_tensor(intrinsics_vec)
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        #image = img[:h-h%16, :w-w%16]

        queue.put((t, images.squeeze(0), intrinsics))

    queue.put((-1, images.squeeze(0), intrinsics))



# def image_ivm_300_stream(queue, imagedir, stride, skip=0):
#     """ image generator """
#
#     img_exts = ["*.png", "*.jpeg", "*.jpg", "*.JPG"]
#
#     print("imagedir : ", imagedir)
#
# #    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
#
#     image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))
#
#
#
#     print("image_list : ", image_list)
#
#     print("images lenght : ", len(image_list))
#
#
#
#     for t, imfile in enumerate(image_list):
#         image = cv2.imread(str(imfile))
#
#        
#         print("imfile : ", imfile)
#
#         h, w, _ = image.shape
#
#         img = image
#         H, W, _ = img.shape
#         
#         # rectification
#         K_r = np.array([
#             322.638671875, 0, 255.9466552734375,
#             0, 322.638671875, 187.4475402832031,
#             0, 0, 1
#             ]).reshape(3,3)
#         d_r = np.array([
#             -0.070313379,
#             0.071827024,
#             0.0004486586,
#             0.00070285366,
#             -0.015095583
#             ]).reshape(5)
#         R_r = np.array([
#             0.9999984896881986, -0.001713768657967563, -0.0002891683050380818,
#             0.001714143276046202, 0.9999976855080072, 0.001300265918105914,
#             0.0002869392807828858, -0.001300759630204676, 0.9999991128447232
#             ]).reshape(3,3)
#
#         P_r = np.array([
#             322.6092376708984, 0, 257.7363166809082, 48.37263543147446,
#             0, 322.6092376708984, 186.6225147247314, 0,
#             0, 0, 1, 0]).reshape(3,4)
#
#         map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (W, H), cv2.CV_32F)
#
#         intrinsics_vec = [322.6092376708984, 322.6092376708984, 257.7363166809082, 186.6225147247314]
#
#         ht0, wd0 = [376, 514]
#
#         images = [cv2.remap(img, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
#
#         images = torch.from_numpy(np.stack(images, 0))
#
#         image_tmp = images.numpy().squeeze(0)
#         # cv2.imshow("title", image_tmp)
#         # cv2.waitKey(0)
#  
#         #images = images.permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
#         images = images.permute(0, 3, 1, 2)
#
#         # Ensure either size or scale_factor is defined
#
#         #image_size = [448, 736]
#         image_size = [528, 960]
#         #image_size = [514, 616]
#
#         #print("++++++++ image_size  : ",image_size)
#         if image_size is not None:
#             images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
#         else:
#             raise ValueError("image_size must be defined")
#             
#         intrinsics = torch.as_tensor(intrinsics_vec)
#         intrinsics[0] *= image_size[1] / wd0
#         intrinsics[1] *= image_size[0] / ht0
#         intrinsics[2] *= image_size[1] / wd0
#         intrinsics[3] *= image_size[0] / ht0
#
#         #image = img[:h-h%16, :w-w%16]
#
#         queue.put((t, images.squeeze(0), intrinsics))
#
#     queue.put((-1, images.squeeze(0), intrinsics))




def image_ivm_300_stream(queue, imagedir, stride, skip=0):
    """ image generator """

    img_exts = ["*.png", "*.jpeg", "*.jpg", "*.JPG"]

    print("imagedir : ", imagedir)

#    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]

    image_list = sorted(glob.glob(os.path.join(imagedir, 'left', '*.JPG')))[::stride]
    #depth_list = sorted(glob.glob(os.path.join(datapath, 'depth', '*.png')))[::stride]
    depth_list = sorted(glob.glob(os.path.join(imagedir, 'depth', '*.npy')))[::stride]

    print("image_list : ", image_list[:10])
    print("depth_list : ", depth_list[:10])


    #for t, imfile in enumerate(image_list):

    for t, (imfile, depth_file) in enumerate(zip(image_list, depth_list)):
        image = cv2.imread(str(imfile))

       
        print("imfile : ", imfile)

        h, w, _ = image.shape

        img = image
        H, W, _ = img.shape
        
        # rectification
        K_r = np.array([
            322.638671875, 0, 255.9466552734375,
            0, 322.638671875, 187.4475402832031,
            0, 0, 1
            ]).reshape(3,3)
        d_r = np.array([
            -0.070313379,
            0.071827024,
            0.0004486586,
            0.00070285366,
            -0.015095583
            ]).reshape(5)
        R_r = np.array([
            0.9999984896881986, -0.001713768657967563, -0.0002891683050380818,
            0.001714143276046202, 0.9999976855080072, 0.001300265918105914,
            0.0002869392807828858, -0.001300759630204676, 0.9999991128447232
            ]).reshape(3,3)

        P_r = np.array([
            322.6092376708984, 0, 257.7363166809082, 48.37263543147446,
            0, 322.6092376708984, 186.6225147247314, 0,
            0, 0, 1, 0]).reshape(3,4)

        map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (W, H), cv2.CV_32F)

        intrinsics_vec = [322.6092376708984, 322.6092376708984, 257.7363166809082, 186.6225147247314]

        ht0, wd0 = [376, 514]

        images = [cv2.remap(img, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]

        images = torch.from_numpy(np.stack(images, 0))

        image_tmp = images.numpy().squeeze(0)
        # cv2.imshow("title", image_tmp)
        # cv2.waitKey(0)
 
        #images = images.permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
        images = images.permute(0, 3, 1, 2)

        # Ensure either size or scale_factor is defined

        #image_size = [448, 736]
        image_size = [528, 960]
        #image_size = [514, 616]

        #print("++++++++ image_size  : ",image_size)
        if image_size is not None:
            images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
        else:
            raise ValueError("image_size must be defined")
            
        intrinsics = torch.as_tensor(intrinsics_vec)
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        depth = -np.load(depth_file)/1000
        # print("depth values : ")
        # print("depth min : ",np.min(depth))
        # print("depth max : ",np.max(depth))
        # print("depth shape : ",depth.shape)
        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None,None], image_size).squeeze()

        # disps_sens
        disp_sens = torch.where(depth>0, 1.0/depth, depth)
        #image = img[:h-h%16, :w-w%16]

        queue.put((t, images.squeeze(0), disp_sens, intrinsics))

    queue.put((-1, images.squeeze(0), disp_sens, intrinsics))









def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    assert os.path.exists(imagedir), imagedir
    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

