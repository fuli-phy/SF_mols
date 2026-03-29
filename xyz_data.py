from ase.io import read
import torch
import numpy as np
from pymatgen.core import Molecule
from itertools import chain
from ase.neighborlist import NeighborList

#struc = read(r'G:\work\24_Singlet_fission\FORMED_database\XYZ_FORMED\XYZ_FORMED\AAXTHP.xyz')
#struc = read('CID_241.xyz')

def get_atom_position(struc):
    """get atom positions with size [num_atom, 3]"""
    pos = struc.positions
    return torch.tensor(pos)

def get_atom_element(struc):
    ele_list = []
    for atoms in struc:
        ele_list.append(atoms.symbol)
    return ele_list

def get_edge_index(struc, R_cut):
    """ get edge_index from a structure with a cutoff R_CUT
    return the edge_matrix with shape [2,num_edge]"""
    dist_matrix = torch.tensor(struc.get_all_distances(struc))
    adj_matrix = dist_matrix.lt(R_cut).int()
    edge_matrix = torch.nonzero(adj_matrix-torch.eye(adj_matrix.size()[0],adj_matrix.size()[1])) # double count
#    edge_matrix = torch.nonzero(adj_matrix*torch.triu(torch.ones(dist_matrix.size()),diagonal=1)) # single count
    edge_index = edge_matrix.transpose(0,1)
    return edge_index

def get_atom_neighbors(filename, R_cut):
    "get list of center atom's neighbors, return N * Nneig list"
    stru = Molecule.from_file(filename)
    total_neighbors_list = []
    for i in range(stru.num_sites):
        neighbors = stru.get_neighbors(stru[i], R_cut)
        neighbors_ele = [neighbors[site].specie.symbol for site in range(len(neighbors))]
        total_neighbors_list.append(neighbors_ele)
    return total_neighbors_list


def get_edge_vector(struc, edge_index):
    edge_vector = []
    for i in range(edge_index.shape[-1]):
        edge_vector.append(struc[int(edge_index[0][i])].position - struc[int(edge_index[1][i])].position)
    return torch.tensor(np.array(edge_vector))


def get_neig_id(edge_index, id):
    neig_id =  [i for i in range(len(edge_index[0])) if edge_index[0][i] == id]
    return edge_index[1][neig_id].tolist()

def get_hole_atoms(filename):
    stru = read(filename)
    total_neighbors_list = get_atom_neighbors(filename, 2)
    edge_index = get_edge_index(stru, 2)
    B_edge_feature1 = ['B', 'B', 'B', 'B', 'B']
    B_edge_feature2 = ['B', 'B', 'B', 'B']
    B_edge_feature3 = ['B', 'B', 'H']
    #B_edge_list = [id for id in range(len(total_neighbors_list)) if total_neighbors_list[id] == B_edge_feature1]
    B_edge_list = [id for id in range(len(total_neighbors_list)) if total_neighbors_list[id]==B_edge_feature1
                   or total_neighbors_list[id]==B_edge_feature2 or total_neighbors_list[id]==B_edge_feature3]
    print(B_edge_list)
    ring_list = []
    count = 0

    for one in B_edge_list:
        if len(ring_list) == 0:
            sum_list = []
        else:
            sum_list = list(set(chain.from_iterable(list(ring_list))))
        if one in sum_list:
            continue
        two_ = list(set(B_edge_list) & set(get_neig_id(edge_index, one)))
        if len(two_) == 1:
            continue
        for two in two_:
            if len(ring_list) == 0:
                sum_list = []
            else:
                sum_list = list(set(chain.from_iterable(list(ring_list))))
            if two in sum_list:
                continue
            three_ = list(set(B_edge_list) & set(get_neig_id(edge_index, two)))
            if len(three_) <= 1:
                continue
            if one in three_:
                three_.remove(one)
            for three in three_:
                if len(list(set(get_neig_id(edge_index, three)) & set(get_neig_id(edge_index, one)))) == 2:
                    continue
                if three in sum_list:
                    continue
                four_ = list(set(B_edge_list) & set(get_neig_id(edge_index, three)))
                if len(four_) <= 1:
                    continue
                if two in four_:
                    four_.remove(two)
                for four in four_:
                    if len(list(set(get_neig_id(edge_index, four)) & set(get_neig_id(edge_index, two)))) == 2:
                        continue
                    if four in sum_list:
                        continue
                    five_ = list(set(B_edge_list) & set(get_neig_id(edge_index, four)))
                    if len(five_) <= 1:
                        continue
                    if three in five_:
                        five_.remove(three)
                    for five in five_:
                        if len(list(set(get_neig_id(edge_index, five)) & set(get_neig_id(edge_index, three)))) == 2:
                            continue
                        if five in sum_list:
                            continue
                        six_ = list(set(B_edge_list) & set(get_neig_id(edge_index, five)))
                        if len(six_) <= 1:
                            continue
                        if four in six_:
                            six_.remove(four)
                        for end in six_:
                            if end in sum_list:
                                continue
                            if one in get_neig_id(edge_index, end):
                                ring = [one, two, three, four, five, end]
                                ring = sorted(ring)
                                if ring not in ring_list:
                                    count += 1
                                    ring_list.append(ring)
    return ring_list


def get_ring_center(struc, ring_list):
    center_pos_list = []
    for i in range(len(ring_list)):
        temp = [struc[int(ring_list[i][j])].position for j in range(len(ring_list[i]))]
        temp_ = np.sum(temp, axis=0).tolist()
        center_pos = [x / len(ring_list[i]) for x in temp_]
        center_pos_list.append(center_pos)
    return center_pos_list




#edge_vector(struc, 1)


#adjacency_matrix(struc, 1.5)


#edge_vector = get_edge_vector(struc,edge_index)
#print(edge_vector.shape)

#get_atom_neighbors('1.xyz', 2)

#ring_list = get_hole_atoms('2.xyz')
#struc = read('2.xyz')
#ring_center = get_ring_center(struc, ring_list)
#print(ring_center)
#print(len(ring_center))
