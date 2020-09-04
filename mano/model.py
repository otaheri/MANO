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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from collections import namedtuple

import torch
import torch.nn as nn

from .lbs import lbs
from .utils import Struct, to_np, to_tensor
from .utils import Mesh,points2sphere, colors
from .joints_info import TIP_IDS

ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'full_pose', 'betas',
                          'global_orient',
                          'hand_pose'
                          ])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def load(model_path, is_rhand=True, **kwargs):
    ''' load MANO model from a path

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- mano
                |-- MANO_RIGHT
                |-- MANO_LEFT
        is_rhand: str, optional
            When model_path is the mano folder, then this parameter specifies  the
            left or right of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            hand_model: nn.Module
                The PyTorch module that implements the corresponding hand model
    '''

    return MANO(model_path, is_rhand, **kwargs)

class MANO(nn.Module):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS
    NUM_BETAS = 10

    def __init__(self,
                 model_path,
                 is_rhand=True,
                 data_struct=None,
                 create_betas=True,
                 betas=None,
                 create_global_orient=True,
                 global_orient=None,
                 create_transl=True,
                 transl=None,
                 create_hand_pose=True,
                 hand_pose=None,
                 use_pca=True,
                 num_pca_comps=6,
                 flat_hand_mean=False,
                 batch_size=1,
                 joint_mapper=None,
                 v_template=None,
                 dtype=torch.float32,
                 vertex_ids=None,
                 use_compressed=True,
                 ext='pkl',
                 **kwargs):
        ''' MANO model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            create_hand_pose: bool, optional
                Flag for creating a member variable for the pose of the
                hand. (default = True)
            hand_pose: torch.tensor, optional, BxP
                The default value for the left hand pose member variable.
                (default = None)
            num_pca_comps: int, optional
                The number of PCA components to use for each hand.
                (default = 6)
            flat_hand_mean: bool, optional
                If False, then the pose of the hand is initialized to False.
            batch_size: int, optional
                The batch size used for creating the member variables
            dtype: torch.dtype, optional
                The data type for the created variables
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.num_pca_comps = num_pca_comps
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format('RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = True if 'RIGHT' in os.path.basename(model_path) else False
            assert osp.exists(mano_path), 'Path {} does not exist!'.format(
                mano_path)

            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)


        self.tip_ids = TIP_IDS['mano']

        super(MANO, self).__init__()

        self.batch_size = batch_size
        self.dtype = dtype
        self.joint_mapper = joint_mapper

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        if create_betas:
            if betas is None:
                default_betas = torch.zeros([batch_size, self.NUM_BETAS],
                                            dtype=dtype)
            else:
                if 'torch.Tensor' in str(type(betas)):
                    default_betas = betas.clone().detach()
                else:
                    default_betas = torch.tensor(betas,
                                                 dtype=dtype)

            self.register_parameter('betas', nn.Parameter(default_betas,
                                                          requires_grad=True))

        if create_global_orient:
            if global_orient is None:
                default_global_orient = torch.zeros([batch_size, 3],
                                                    dtype=dtype)
            else:
                if torch.is_tensor(global_orient):
                    default_global_orient = global_orient.clone().detach()
                else:
                    default_global_orient = torch.tensor(global_orient,
                                                         dtype=dtype)

            global_orient = nn.Parameter(default_global_orient,requires_grad=True)
            self.register_parameter('global_orient', global_orient)

        if create_transl:
            if transl is None:
                default_transl = torch.zeros([batch_size, 3], dtype=dtype, requires_grad=True)
            else:
                default_transl = torch.tensor(transl, dtype=dtype)
            self.register_parameter('transl', nn.Parameter(default_transl, requires_grad=True))


        if v_template is None:
            v_template = data_struct.v_template
        if not torch.is_tensor(v_template):
            v_template = to_tensor(to_np(v_template), dtype=dtype)
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(data_struct.v_template), dtype=dtype))

        # The shape components
        shapedirs = data_struct.shapedirs
        # The shape components
        self.register_buffer('shapedirs', to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',to_tensor(to_np(data_struct.weights), dtype=dtype))

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps ==45:
            self.use_pca = False
        self.flat_hand_mean = flat_hand_mean

        hand_components = data_struct.hands_components[:num_pca_comps]

        self.np_hand_components = hand_components

        if self.use_pca:
            self.register_buffer(
                'hand_components',
                torch.tensor(hand_components, dtype=dtype))

        if self.flat_hand_mean:
            hand_mean = np.zeros_like(data_struct.hands_mean)
        else:
            hand_mean = data_struct.hands_mean

        self.register_buffer('hand_mean',
                             to_tensor(hand_mean, dtype=self.dtype))

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_hand_pose:
            if hand_pose is None:
                default_hand_pose = torch.zeros([batch_size, hand_pose_dim],
                                                dtype=dtype)
            else:
                default_hand_pose = torch.tensor(hand_pose, dtype=dtype)

            hand_pose_param = nn.Parameter(default_hand_pose,
                                           requires_grad=True)
            self.register_parameter('hand_pose', hand_pose_param)

        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose(data_struct,
                                          flat_hand_mean=flat_hand_mean)
        pose_mean_tensor = pose_mean.clone().to(dtype)
        self.register_buffer('pose_mean', pose_mean_tensor)

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)

        pose_mean = torch.cat([global_orient_mean,
                               self.hand_mean], dim=0)
        return pose_mean

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def extra_repr(self):
        msg = 'Number of betas: {}'.format(self.NUM_BETAS)
        if self.use_pca:
            msg += '\nNumber of PCA components: {}'.format(self.num_pca_comps)
        msg += '\nFlat hand mean: {}'.format(self.flat_hand_mean)
        return msg

    def add_joints(self,vertices,joints, joint_ids = None):

        if joint_ids is None:
            joint_ids = to_tensor(list(self.tip_ids.values()),
                                  dtype=torch.long)
        extra_joints = torch.index_select(vertices, 1, joint_ids)
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints


    def forward(self, betas=None, global_orient=None, hand_pose=None, transl=None,
                return_verts=True, return_tips = False, return_full_pose=False, pose2rot=True,
                **kwargs):
        '''
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        betas = betas if betas is not None else self.betas
        hand_pose = (hand_pose if hand_pose is not None else self.hand_pose)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum('bi,ij->bj', [hand_pose, self.hand_components])

        full_pose = torch.cat([global_orient,
                               hand_pose], dim=1)
        full_pose += self.pose_mean

        if return_verts:
            vertices, joints = lbs(self.betas, full_pose, self.v_template,
                                   self.shapedirs, self.posedirs,
                                   self.J_regressor, self.parents,
                                   self.lbs_weights, pose2rot=pose2rot,
                                   dtype=self.dtype)

            # Add any extra joints that might be needed
            if return_tips:
                joints = self.add_joints(vertices, joints)

            if self.joint_mapper is not None:
                joints = self.joint_mapper(joints)

            if apply_trans:
                joints = joints + transl.unsqueeze(dim=1)
                vertices = vertices + transl.unsqueeze(dim=1)

        output = ModelOutput(vertices=vertices if return_verts else None,
                             joints=joints if return_verts else None,
                             betas=betas,
                             global_orient=global_orient,
                             hand_pose = hand_pose,
                             full_pose=full_pose if return_full_pose else None)

        return output

    def hand_meshes(self,output, vc= colors['skin']):

        vertices = to_np(output.vertices)
        if vertices.ndim <3:
            vertices = vertices.reshape(-1,778,3)

        meshes = []
        for v in vertices:
            hand_mesh = Mesh(vertices=v, faces=self.faces, vc=vc)
            hand_mesh.visual.vertex_colors[:,3] = 164
            meshes.append(hand_mesh)

        return  meshes

    def joint_meshes(self,output, radius=.002, vc=colors['green']):

        joints = to_np(output.joints)
        if joints.ndim <3:
            joints = joints.reshape(1,-1,3)

        meshes = []
        for j in joints:
            joint_mesh = Mesh(vertices=j, radius=radius, vc=vc)
            meshes.append(joint_mesh)

        return  meshes