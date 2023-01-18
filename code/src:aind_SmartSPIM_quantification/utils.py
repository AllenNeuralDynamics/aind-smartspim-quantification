#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:30:28 2023

@author: nicholas.lusk
"""

import copy
import json

import numpy as np

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

CCF_mesh_dir = str('/allen/programs/celltypes/workgroups/mct-t200/Molecular_Genetics_Daigle_Team/Elyse/CCF_mesh/CCF_meshes/json_verts_float/')

# mcc orientation is [AP, DV, ML(left -> right)]
mcc = MouseConnectivityCache(resolution=25)
refspace = mcc.get_reference_space()
acrnm_map = refspace.structure_tree.get_id_acronym_map() # dictionary mapping acronyms to ids
name_map = {v: k for k, v in acrnm_map.items()}

def get_CCF_mesh_points(structure_id, CCF_mesh_dir):
    with open('{}{}.json'.format(CCF_mesh_dir, structure_id)) as f:
        structure_data = json.loads(f.read())
        vertices,faces = (np.array(structure_data[str(structure_id)]['vertices']),
                          np.array(structure_data[str(structure_id)]['faces']))
    return vertices,faces

def reflect_about_midline(vertices):
    vertices_right = copy.copy(vertices)
    vertices_right[:,0] = vertices_right[:,0]+(5700-vertices_right[:,0])*2
    return vertices_right