import dgl
import torch
from torch.nn import GroupNorm
from torch import nn
import torch.nn.functional as F
from CASTLE.Model.gpaw_encoder import GPAW
from torch_geometric.nn import TransformerConv

class Atom_GTN(torch.nn.Module):
    def __init__(self, num_feature, e_dim, out_dim, heads, dropout):
        super(Atom_GTN, self).__init__()
        ARMAlayer = 3
        self.conv1 = TransformerConv(num_feature, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.conv2 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)
        self.conv3 = TransformerConv(out_dim, out_dim, heads, edge_dim=e_dim, dropout=dropout, concat=False)

        self.gn = GroupNorm(4, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr):
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.ln(x)
        x = F.gelu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.ln(x)
        x = F.gelu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.ln(x)
        x = F.gelu(x)
        return x

#################################### Seqfeatures #####################################
class seq_difference(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(seq_difference, self).__init__()
        self.linear1 = nn.Linear(in_dim*2, in_dim*1)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.layernorm1 = nn.LayerNorm(in_dim)
        self.layernorm2 = nn.LayerNorm(out_dim)

    def forward(self, seq1, seq2):
        # x = torch.sub(seq1,seq2)
        x = torch.cat([seq1,seq2],dim=0)
        x = self.linear1(x)
        x = self.layernorm1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = self.layernorm2(x)
        x = F.gelu(x)
        return x

#########################################################################################
struct_dim = 512
local_metal_dim = 512
atom_dim = 256
common_dim = 512
#no_seq

#################################### Model #####################################
class Model_Net(nn.Module):
    def __init__(self, metal_dim=63,in_dim=1048, out_dim=512, atom_heads=8, atom_drop=0.5):
        super(Model_Net, self).__init__()
        self.seq_difference = seq_difference(in_dim, out_dim)
        self.atom_Encoder = Atom_GTN(65, 3, atom_dim, atom_heads, atom_drop)
        print('Atom Gtn:', atom_dim, 'atom_heads:', atom_heads, 'atom_drop:', atom_drop)

        self.dmpnn = DMPNN()
        # 将v1投影到common_dim
        self.project_v1 = nn.Linear(struct_dim * 2, common_dim)
        # 将v2投影到common_dim
        self.project_v2 = nn.Linear(out_dim+metal_dim, common_dim)
        # 定义门控网络
        self.gate = nn.Linear(common_dim * 2, 1)

        layer_dims = [512, 256, 128, 64, 32, 16, 1]
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        current_dim = out_dim
        for dim in layer_dims:
            self.layers.append(nn.Linear(current_dim, dim))
            self.norm_layers.append(nn.LayerNorm(dim))
            current_dim = dim

    def get_pos_edge(self,g):
        node_pos = g.ndata["node_coordinate"]
        row, col = g.edges()
        row_pos = node_pos[row]
        col_pos = node_pos[col]
        return row_pos-col_pos

    def forward(self, res_node_from,res_index_from, res_edge_from, atom2res_from,atom_node_from,atom_index_from, atom_edge_from,node_pos_from,basic_attn_from,seq_bm, seq_am, metal,devices):

        atom_node_from = self.atom_Encoder(atom_node_from, atom_index_from, atom_edge_from)
        g = dgl.graph((torch.cat((res_index_from[0],res_index_from[1]),dim=0),torch.cat((res_index_from[1],res_index_from[0]),dim=0)))
        g.ndata["res_feature_h"] = res_node_from
        g.ndata["res_index_atom"] = atom2res_from
        g.ndata["node_coordinate"] = node_pos_from
        g.edata["edge_feature_h"] = torch.cat((res_edge_from,res_edge_from),dim=0)
        g.edata["edge_pos_coordinate"] = self.get_pos_edge(g)
        g.edata['basic_attn'] = torch.cat((basic_attn_from,basic_attn_from),dim=0)
        g.to(device=devices)

        struct_metal_fea = self.dmpnn(g,atom_node_from,devices)
        seq = self.seq_difference(seq_bm, seq_am)
        v1_proj = self.project_v1(struct_metal_fea)
        v2_proj = self.project_v2(torch.cat((seq, metal), dim=0))

        combined = torch.cat((v1_proj, v2_proj), dim=0)
        gate_output = torch.sigmoid(self.gate(combined))
        x = gate_output * v1_proj + (1 - gate_output) * v2_proj

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.norm_layers[i](x)
            x = F.gelu(x)
        x = self.layers[-1](x)
        x = torch.sigmoid(x)
        return x









