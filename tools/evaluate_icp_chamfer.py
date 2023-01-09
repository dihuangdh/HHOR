'''
Date: 2022-01-26 5:25:23 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2023-01-08 8:51:22 pm
LastEditors: dihuangdh
'''

import os
import trimesh
import numpy as np
import pyrender
import copy
import torch
from tqdm import tqdm

from icp import mesh_other
from chamferdist import ChamferDistance


def pyrender_vis(meshes):  # press 'i' to toggle axis display
    scene = pyrender.Scene()
    for mesh in meshes:
        scene.add(pyrender.Mesh.from_trimesh(mesh))
    pyrender.Viewer(scene, use_raymond_lighting=True)


def register(source, target, type='icp_common'):
    if type == 'icp_common':
        from trimesh.registration import mesh_other
    elif type == 'icp_constrained':
        from icp import mesh_other
    else:
        raise ValueError('Registration Type Should Be in {icp_common} and {icp_constrained}.')

    # register
    source2target, cost = mesh_other(source, target, scale=True)
    # source2target, cost = mesh_other(source, target, scale=False)

    # transform
    source.apply_transform(source2target)

    return source, target



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/home/dihuang/data/mesh/source.ply')
    parser.add_argument('--target', type=str, default='/home/dihuang/data/mesh/target.ply')
    parser.add_argument('--flip_x', action='store_true')
    parser.add_argument('--flip_y', action='store_true')
    parser.add_argument('--flip_z', action='store_true')
    parser.add_argument('--iter', type=int, default=1)
    args = parser.parse_args()

    chamferDist = ChamferDistance()

    source = trimesh.load(args.source)  # reconstructed mesh
    target = trimesh.load(args.target)  # ground truth mesh

    # flip
    if args.flip_x:
        source.apply_translation([-source.centroid[0], 0, 0])
    if args.flip_y:
        source.apply_translation([0, -source.centroid[1], 0])
    if args.flip_z:
        source.apply_translation([0, 0, -source.centroid[2]])

    # normalize
    source.vertices -= source.center_mass
    source.vertices /= source.vertices.max()
    target.vertices -= target.center_mass
    target.vertices /= target.vertices.max()

    for i in range(args.iter):

        # register
        new_source, _ = register(source, target)
        # new_source = source
        # if args.iter == 1:
        #     pyrender_vis([new_source, target])

        # chamfer distance 
        vertices_source = new_source.vertices
        vertices_target = target.vertices
        vertices_source = torch.tensor(vertices_source, dtype=torch.float32)[None].cuda()
        vertices_target = torch.tensor(vertices_target, dtype=torch.float32)[None].cuda()
        dist_bidirectional = chamferDist(vertices_source, vertices_target, bidirectional=True) * 0.001
        print(dist_bidirectional.detach().cpu().item())
