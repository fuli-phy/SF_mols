import xyz_data
import torch
from ase.io import read
import os
import pandas as pd



def read_csv_file(csv_file):
    print(pd.read_csv(csv_file).head())

def get_name_properties(xyz_folder, csv_file):
    pos_list, ele_list, edge_index_list, edge_vector_list = [], [], [], []
    s1_list, t1_list, homo_list, lumo_list, gap_list = [], [], [], [], []
    fs1_list = []
    data = pd.read_csv(csv_file)
    for i, xyz_name in enumerate(data['name']):
        print(xyz_name)
        xyz_file = os.path.join(xyz_folder, str(data['name'][i]+'.xyz'))
        s1, s2, s3, s4, s5 = data['S1_exc'][i], data['S2_exc'][i],data['S3_exc'][i],data['S4_exc'][i],data['S5_exc'][i]
        t1, t2, t3, t4, t5 = data['T1_exc'][i], data['T2_exc'][i],data['T3_exc'][i],data['T4_exc'][i],data['T5_exc'][i]
        homo, lumo, gap = data['HOMO'][i], data['LUMO'][i], data['gap'][i]
        fs1 = data['S1_osc'][i]
        struc = read(xyz_file)
        pos = xyz_data.get_atom_position(struc)
        ele = xyz_data.get_atom_element(struc)
        edge_index = xyz_data.get_edge_index(struc, 3)
        edge_vector = xyz_data.get_edge_vector(struc,edge_index)
        pos_list.append(pos)
        ele_list.append(ele)
        edge_index_list.append(edge_index)
        edge_vector_list.append(edge_vector)
        s1_list.append(s1)
        t1_list.append(t1)
        homo_list.append(homo)
        lumo_list.append(lumo)
        gap_list.append(gap)
        fs1_list.append(fs1)
    return pos_list, ele_list, edge_index_list, edge_vector_list, s1_list, t1_list, homo_list, lumo_list, gap_list, fs1_list
    #return fs1_list

def data_save(properties, name):
    pt_filename = str(str(name) + '.pt')
    torch.save(properties, pt_filename)

def load_data(pt_file):
    loaded_data = torch.load(pt_file)
    print(loaded_data[0])
    print(len(loaded_data))


if __name__ == '__main__':
    csv_file = '/data/home/fuli/work/GPU5_fuli/24_Singlet_fission/xyz_file/Data_FORMED_scored.csv'
    xyz_folder = '/data/home/fuli/work/GPU5_fuli/24_Singlet_fission/xyz_file/XYZ_FORMED'
    #csv_file = r'G:\work\24_Singlet_fission\FORMED_database\Data_FORMED_scored.csv'
    #xyz_folder = r'G:\work\24_Singlet_fission\FORMED_database\XYZ_FORMED\XYZ_FORMED'
    pos, ele, edge_index, edge_vector, s1, t1, homo, lumo, gap, fs1 = get_name_properties(xyz_folder, csv_file)
    data_save(pos, 'pos')
    data_save(ele, 'ele')
    data_save(edge_index, 'edge_index')
    data_save(edge_vector, 'edge_vector')
    data_save(s1, 's1')
#    data_save(t1, 't1')
#    data_save(homo, 'homo')
#    data_save(lumo, 'lumo')
#    data_save(gap, 'gap')
#    data_save(fs1, 'fs1')

    #load_data(r'E:\Work\Project\GNN\GNN_singlet_fission_Fu\singlet_fission_dataset_cutoff4\edge_index.pt')

