'''
Date: 2021-11-05 6:54:00 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2022-08-10 2:25:19 pm
LastEditors: dihuangdh
'''
import os
from tqdm import tqdm
import cv2
import json
import numpy as np
from easymocap.mytools.camera_utils import read_camera
from easymocap.smplmodel import load_model
from read_write_model import read_model, Image, Camera, Point3D, write_model, rotmat2qvec
import shutil
import trimesh


def read_json(jsonname):
    with open(jsonname) as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='')
    parser.add_argument('--step', type=int, default=1)
    args = parser.parse_args()

    imgs_folder = os.path.join(args.data_dir, 'images')
    masks_folder = os.path.join(args.data_dir, 'masks')
    subs = os.listdir(imgs_folder)
    if len(subs) == 1:
        smpl_folder = os.path.join(args.data_dir, 'output', 'smpl')
    else:
        smpl_folder = os.path.join(args.data_dir, 'output', 'smpl')
    fused_dir = os.path.join(args.data_dir, 'colmap', 'images')
    npz_dir = os.path.join(args.data_dir, 'colmap', 'semantic_maps')
    png_dir = os.path.join(args.data_dir, 'colmap', 'images_png')
    os.makedirs(fused_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    # read cameras
    print('Reading cameras...')
    intri_path = os.path.join(args.data_dir, 'intri.yml')
    extri_path = os.path.join(args.data_dir, 'extri.yml')
    cams = read_camera(intri_path, extri_path)

    # mano model
    print('Initialize model...')
    body_model = load_model(gender='neutral', model_type='manol', model_path='EasyMocap/data/smplx')

    points3D_out = {}
    images_out = {}
    cameras_out = {}
    key, img_id, camera_id = 0, 0, 0
    xys_ = np.zeros((0, 2), float)
    point3D_ids_ = np.full(0, -1, int)
    vertices_all = []

    for sub in subs:
        print(f'Processing sub: {sub}')

        # camera
        cam = cams[sub]
        K = cam['K']  # (3, 3)
        R = cam['R']  # (3, 3)
        T = cam['T']  # (3, 1)

        img_folder = os.path.join(imgs_folder, sub)
        mask_folder = os.path.join(masks_folder, sub)
        imgnames = sorted(os.listdir(img_folder))
        imgnames = imgnames[::args.step]
        for idx, imgname in enumerate(tqdm(imgnames)):

            key += 1
            img_id += 1
            camera_id += 1

            # read image
            imgpath = os.path.join(img_folder, imgname)
            img = cv2.imread(imgpath)
            img_h, img_w = img.shape[0], img.shape[1]

            smpl_name = imgname.replace('.jpg', '.json')
            smpl_data = read_json(os.path.join(smpl_folder, smpl_name))[0]
            vertices=body_model(return_verts=True, return_tensor=False, **smpl_data)[0]  # (778, 3)

            # # project
            # vert_3d = vertices.copy()
            # Rt = np.concatenate([R, T], 1)  # (3, 4)
            # vert_2d = np.concatenate([vert_3d, np.ones((vert_3d.shape[0], 1))], 1)
            # vert_2d = Rt @ vert_2d.transpose(1, 0)
            # vert_2d = K @ vert_2d
            # vert_2d = (vert_2d[:2, :] / (vert_2d[2:, :]+1e-5)).transpose(1, 0)
            # for point_2d in vert_2d:
            #     img_viz = cv2.circle(img, (int(point_2d[0]), int(point_2d[1])), radius=4, color=(0, 0, 255), thickness=-1)
            #     img_viz = cv2.resize(img_viz, (0, 0), fx=0.25, fy=0.25)
            # cv2.imshow('image', img_viz)
            # cv2.waitKey(0)
            # import ipdb; ipdb.set_trace()

            Rh = np.array(smpl_data['Rh'])
            Th = np.array(smpl_data['Th'])
            rot, _ = cv2.Rodrigues(Rh)
            Rtmano = np.concatenate((rot, Th[0][:, None]), -1)  # (3, 4)
            Rtmano = np.concatenate((Rtmano, [[0, 0, 0, 1]]), 0)

            smpl_data['Rh'] = np.zeros((1, 3))
            smpl_data['Th'] = np.zeros((1, 3))
            vertices=body_model(return_verts=True, return_tensor=False, **smpl_data)[0]  # (778, 3)

            # # project
            # vert_3d = vertices.copy()
            # Rt = np.concatenate([R, T], 1)  # (3, 4)
            # vert_2d = np.concatenate([vert_3d, np.ones((vert_3d.shape[0], 1))], 1)
            # vert_2d = Rt @ Rtmano @ vert_2d.transpose(1, 0)
            # vert_2d = K @ vert_2d
            # vert_2d = (vert_2d[:2, :] / (vert_2d[2:, :]+1e-5)).transpose(1, 0)
            # for point_2d in vert_2d:
            #     img_viz = cv2.circle(img, (int(point_2d[0]), int(point_2d[1])), radius=4, color=(0, 0, 255), thickness=-1)
            #     img_viz = cv2.resize(img_viz, (0, 0), fx=0.25, fy=0.25)
            # cv2.imshow('image', img_viz)
            # cv2.waitKey(0)
            # import ipdb; ipdb.set_trace()

            Rt = np.concatenate([R, T], 1)  # (3, 4)
            Rtnew = Rt @ Rtmano
            Rnew = Rtnew[:3, :3]
            Tnew = Rtnew[:3, 3:]
            qvec = rotmat2qvec(Rnew)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            image_ = Image(
                id=img_id,
                qvec=qvec,
                tvec=Tnew[:, 0],
                camera_id=camera_id,
                name=imgname,
                xys=xys_,
                point3D_ids=point3D_ids_
            )

            camera_ = Camera(
                id=camera_id,
                model='PINHOLE',
                width=img_w,
                height=img_h,
                params=np.array([fx, fy, cx, cy]),
            )

            images_out[key] = image_
            cameras_out[key] = camera_
            vertices_all.append(vertices.copy())
            
            # print(Rnew)
            # print(Tnew[:, 0])
            # print(vertices)
            # break

            # fuse images
            shutil.copyfile(os.path.join(img_folder, imgname),
                os.path.join(fused_dir, f'{imgname}'))

            # mask
            maskname = imgname.replace('.jpg', '.png')
            mask_npz_name = maskname.replace('.png', '.npz')
            mask_npz_name = f'{mask_npz_name}'
            maskpath = os.path.join(mask_folder, maskname)
            mask = cv2.imread(maskpath, 0)
            mask = mask / 255.
            mask[mask > 0.5] = 1.
            mask[mask < 0.5] = 0.
            mask = mask.astype(np.int8)
            np.savez(os.path.join(npz_dir, mask_npz_name), mask)

            img_png = np.concatenate([img, mask[..., None]*255], -1)
            png_name = imgname.replace('.jpg', '.png')
            cv2.imwrite(os.path.join(png_dir, png_name), img_png)
            # import ipdb; ipdb.set_trace()
    
    vertices_all = np.array(vertices_all)
    vertices = vertices_all.mean(0)  # (778, 3)
    for idx in tqdm(range(vertices.shape[0]), desc='points3D'):
        point_ = Point3D(
            id=idx+1,
            xyz=vertices[idx],
            rgb=np.array([0, 0, 0]),
            error=0,
            image_ids=np.array([0]),
            point2D_idxs=np.array([0])
        )

        points3D_out[idx+1] = point_
        # break
    
    SCENE_ORIGIN = np.mean(vertices, 0)
    SCENE_RADIUS = np.linalg.norm(vertices - SCENE_ORIGIN, axis=1).max()
    print(f'SCENE_ORIGIN: {SCENE_ORIGIN}.')
    print(f'SCENE_RADIUS: {SCENE_RADIUS}')

    os.makedirs(os.path.join(args.data_dir, 'colmap'), exist_ok=True)
    write_model(cameras=cameras_out, images=images_out, points3D=points3D_out, 
        path=os.path.join(args.data_dir, 'colmap'), ext='.bin')
    
    # vertices = (vertices - SCENE_ORIGIN) / (SCENE_RADIUS * 2)
    mesh = trimesh.Trimesh(vertices, body_model.faces)
    mesh.export(os.path.join(args.data_dir, 'colmap', 'mano.ply'))