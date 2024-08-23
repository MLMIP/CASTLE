import os

from Bio import PDB

pdb_dir = 'SMILE/Data/pdb'
if not os.path.exists(pdb_dir):
    os.makedirs(pdb_dir)

def change_three_to_one(res):
    amino_acid = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'PRO': 'P', 'PHE': 'F', 'TYR': 'Y',
                  'TRP': 'W', 'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'ASN': 'N', 'GLN': 'Q', 'ASP': 'D',
                  'GLU': 'E', 'LYS': 'K', 'ARG': 'R', 'HIS': 'H',
                  # 'HOH': '',  # 标准氨基酸
                  # 'MSE': 'M', 'PTR': 'Y', 'TYS': 'Y', 'SEP': 'S', 'TPO': 'T', 'HIP': 'H'  # 非标准氨基酸
                  }
    if res in amino_acid:
        res = amino_acid[res]
        return res
    else:
        return '_'

class PDBdata(object):
    def __init__(self):
        super(PDBdata,self).__init__()

    def download_clean_pdb(self, pdb_id, chain_id):
        in_pdb_file = os.path.join(pdb_dir, pdb_id.upper() + '.pdb')
        out_pdb_file = os.path.join(pdb_dir, pdb_id + '_' + chain_id + '.pdb')

        if not os.path.exists(out_pdb_file):
            if not os.path.exists(in_pdb_file):
                os.chdir(pdb_dir)
                os.system(f'wget https://files.rcsb.org/view/{pdb_id.upper()}.pdb')
            with open(in_pdb_file, 'r') as f_r, open(out_pdb_file, 'w') as f_w:
                for line in f_r:
                    info = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27],
                            line[30:38],
                            line[38:46], line[46:54]]
                    info = [i.strip() for i in info]
                    if info[5] == chain_id and (info[0] == 'ATOM' or info[0] == 'HETAT'):
                        f_w.write(line)
                    if 'ENDMDL' in line:
                        break
        # if not os.path.exists(f'{pdb_dir}/{pdb_id}.pdb'):
        #     os.system(f'cp {out_pdb_file} {pdb_dir}')

    def read_pdb(self,pdb_id,chain_id,pdb_i,mut_pos,wild_aa,mut_aa):
        PDBdata.download_clean_pdb(self,pdb_id,chain_id)
        pdbfile = pdb_dir + '/' + pdb_id +  '_' + chain_id + '.pdb'
        parser = PDB.PDBParser(QUIET=True)
        struct = parser.get_structure('PDB', pdbfile)
        model = struct[0]
        seq = ''
        position = list()
        chain = model[chain_id]
        for res in chain:
            amino_acid = res.get_resname()
            res_id = res.get_id()
            orig_pos = str(res_id[1]).strip() + str(res_id[2]).strip()
            one_acid = change_three_to_one(amino_acid)
            if one_acid != '_':
                seq += one_acid
                position.append(orig_pos)
        return seq,position


seq, position = PDBdata().read_pdb('3ohm','B','_','_','_','_')
print("fnsdjk")
