# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import trimesh


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todense())
    elif 'chumpy' in str(type(array)):
        array = np.array(array)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array.astype(dtype)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


## vis tools


class Mesh(trimesh.Trimesh):

    def __init__(self,
                 filename=None,
                 vertices=None,
                 faces=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 radius=.002,
                 process = False,
                 visual = None,
                 wireframe=False,
                 smooth = False,
                 **kwargs):

        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process = process)
            vertices = mesh.vertices
            faces= mesh.faces
            visual = mesh.visual
        if vscale is not None:
            vertices = vertices*vscale

        if faces is None:
            mesh = points2sphere(vertices, radius=radius)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rotate_vertices(self,rxyz):
        visual = self.visual
        self.vertices[:] = np.array(self.vertices@rxyz.T)
        self.visual = visual
        return self

    def colors_like(self,color, array, ids):

        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vertex_colors(self,vc, vertex_ids = None):

        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self,fc, face_ids = None):

        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def concatenate_meshes(meshes):
        return trimesh.util.concatenate(meshes)

def points2sphere(points, radius = .001, vc = [0., 0., 1.], count = [5,5]):

    points = points.reshape(-1,3)
    n_points = points.shape[0]

    spheres = []
    for p in range(n_points):
        sphs = trimesh.creation.uv_sphere(radius=radius, count = count)
        sphs.apply_translation(points[p])
        sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)

        spheres.append(sphs)

    spheres = Mesh.concatenate_meshes(spheres)
    return spheres

colors = {
    'pink': [1.00, 0.75, 0.80],
    'skin': [0.96, 0.75, 0.69],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}
