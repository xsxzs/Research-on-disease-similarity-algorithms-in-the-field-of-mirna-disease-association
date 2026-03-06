import torch as th
import torch.nn as nn
from dgl.batch import unbatch
from dgl.transforms import shortest_dist
from torch_geometric.nn import EGConv
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from param import *
from dgl import function as fn
from utils import *
import dgl
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
args = parse_args()
class AvgNeighborSimEncoder(nn.Module):
    def __init__(self, num_rna, num_dis, hidden_dim):
        super(AvgNeighborSimEncoder, self).__init__()
        self.num_rna = num_rna
        self.num_dis = num_dis
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding(num_rna + num_dis, hidden_dim)

    def forward(self, associations, ms, ds):

        rna_neighbor_idx = [[] for _ in range(self.num_rna)]
        dis_neighbor_idx = [[] for _ in range(self.num_dis)]


        for i in range(associations.shape[1]):
            rna_index = associations[0, i]
            dis_index = associations[1, i] - self.num_rna
            rna_neighbor_idx[rna_index].append(dis_index + self.num_rna)
            dis_neighbor_idx[dis_index].append(rna_index)

        neighbor_idx = rna_neighbor_idx + dis_neighbor_idx


        avg_sim = []
        for i in range(len(neighbor_idx)):
            sim = []
            rest = len(neighbor_idx[i])
            for m in range(rest):
                for n in range(m + 1, rest):
                    if i < self.num_rna:
                        temp = ds[neighbor_idx[i][m] - self.num_rna][neighbor_idx[i][n] - self.num_rna]
                    else:
                        temp = ms[neighbor_idx[i][m]][neighbor_idx[i][n]]
                    sim.append(temp)
            sim = np.mean(sim) if sim else 0
            avg_sim.append(sim)

        avg_sim = torch.tensor(avg_sim, dtype=torch.float32)
        indices = (avg_sim * 1000).long()
        neighbor_embedding = self.embedding_layer(indices)
        return neighbor_embedding
class DegreeEncoder(nn.Module):
    def __init__(self, max_degree, embedding_dim):
        super(DegreeEncoder, self).__init__()
        self.encoder1 = nn.Embedding(
            max_degree + 1, embedding_dim, padding_idx=0
        )
        self.encoder2 = nn.Embedding(
            max_degree + 1, embedding_dim, padding_idx=0
        )
        self.max_degree = max_degree

    def forward(self, g):
        in_degree = th.clamp(g.in_degrees(), min=0, max=self.max_degree)
        out_degree = th.clamp(g.out_degrees(), min=0, max=self.max_degree)
        degree_embedding = self.encoder1(in_degree) + self.encoder2(out_degree)
        return degree_embedding
class SpatialEncoder(nn.Module):
    def __init__(self, max_dist, num_heads=1):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        self.embedding_table = nn.Embedding(
            max_dist + 2, num_heads, padding_idx=0
        )
        self.cached_dist = None

    def forward(self, g):
        device = g.device
        g_list = unbatch(g)
        max_num_nodes = th.max(g.batch_num_nodes())
        spatial_encoding = th.zeros(
            len(g_list), max_num_nodes, max_num_nodes, self.num_heads
        ).to(device)
        
        if self.cached_dist is None:
            self.cached_dist = []
            import networkx as nx
            print("Computing all-pairs shortest paths... This may take a minute.")
            for i, ubg in enumerate(g_list):
                ubg_cpu = ubg.cpu()
                dist = (
                        th.clamp(
                            shortest_dist(ubg_cpu, root=None, return_paths=False),
                            min=-1,
                            max=self.max_dist,
                        )
                        + 1
                )
                self.cached_dist.append(dist.to(device))
                
        for i, ubg in enumerate(g_list):
            num_nodes = ubg.num_nodes()
            dist = self.cached_dist[i]
            dist_embedding = self.embedding_table(dist)
            spatial_encoding[i, :num_nodes, :num_nodes] = dist_embedding
        return spatial_encoding
class Linear(nn.Module):
    def __init__(self, args):
        super(Linear, self).__init__()
        n_rna = args.numrna
        n_dis = args.numdis
        self.mf = nn.Linear(n_rna, args.hidden)
        self.df = nn.Linear(n_dis, args.hidden)

    def forward(self, args, m_f, d_f):
        m_f = self.mf(m_f)
        d_f = self.df(d_f)
        return m_f, d_f
class Parallel_GCN(nn.Module):
    def __init__(self, in_feats, out_feats, k=3, method='sum', bias=True, batch_norm=False, dropout=0.0):
        super(Parallel_GCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k + 1
        self.method = method
        self.weights = nn.ParameterList()
        for i in range(self.k):
            self.weights.append(nn.Parameter(th.Tensor(in_feats, out_feats)))
        self.biases = bias
        self.activation = th.relu
        if bias:
            self.biases = nn.ParameterList()
            for i in range(self.k):
                self.biases.append(nn.Parameter(th.Tensor(out_feats)))
        self.reset_parameters()
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)
    def reset_parameters(self):
        for i in range(self.k):
            nn.init.xavier_uniform_(self.weights[i])
            if self.biases is not None:
                nn.init.zeros_(self.biases[i])
    def forward(self, graph, feature, Lambda=0):
        with graph.local_scope():
            degrees = graph.out_degrees().to(feature.device).float().clamp(min=1)
            norm = th.pow(degrees, -0.5)
            shp = norm.shape + (1,) * (feature.dim() - 1)
            norm = th.reshape(norm, shp)
            if self.biases is not None:
                result = th.matmul(feature, self.weights[0]) + self.biases[0]
            else:
                result = th.matmul(feature, self.weights[0])
            for i in range(1, self.k):
                feature = feature * norm
                graph.ndata['h'] = feature
                if 'e' in graph.edata.keys():
                    graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h'))
                else:
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                x = graph.ndata['h'];
                if 'e' in graph.edata:
                    e = graph.edata['e']
                feature = graph.ndata.pop('h')
                feature = feature * norm
                if self.method == 'sum':
                    if self.biases is not None:
                        y = th.matmul(feature, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feature, self.weights[0])
                    result = result + y * (1 - Lambda * i)
                elif self.method == 'mean':
                    if self.biases is not None:
                        y = th.matmul(feature, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feature, self.weights[0])
                    result = result + y * (1 - Lambda * i)
                    result = result / self.k
                elif self.method == 'cat':
                    if self.biases is not None:
                        y = th.matmul(feature, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feature, self.weights[0])
                    result = th.cat((result, y), dim=1)
            if self.batch_norm is not None:
                result = self.batch_norm(result)
            if self.activation is not None:
                result = self.activation(result)
            if self.dropout is not None:
                result = self.dropout(result)
            return result
class Hyperrna(nn.Module):
    def __init__(self, input_size, output_size):
        super(Hyperrna, self).__init__()
        self.inputsize = input_size
        self.outputsize = output_size
        self.conv1 = HypergraphConv(self.inputsize, self.outputsize, use_attention=False, heads=8, concat=False, negative_slope=0.2,
                                    dropout=0.5, bias=True)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)
        self.conv1.apply(init_weights)
    def process_hypergraph(self, triplets, x):
        adj1_matrix = triplets
        adj1_matrix = adj1_matrix.clone().detach()
        hyperedge_index = torch.nonzero(adj1_matrix.T, as_tuple=True)
        hyperedge_index = torch.stack(hyperedge_index)
        hyperedge_index = hyperedge_index.to(x.device)
        output = self.conv1(x, hyperedge_index) # single layer generate the similar result to two-layers
        return output

    def forward(self, X, triplet):
        output = self.process_hypergraph(triplet, X)
        return output
class Hyperdis(nn.Module):
    def __init__(self, input_size, output_size):
        super(Hyperdis, self).__init__()
        self.inputsize = input_size
        self.outputsize = output_size

        self.conv1 = HypergraphConv(self.inputsize, self.outputsize, use_attention=False, heads=8, concat=False, negative_slope=0.2,
                                    dropout=0.5, bias=True)

        # 定义参数初始化函数
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        # 初始化
        self.conv1.apply(init_weights)

    def process_hypergraph(self, triplets, x):
        adj1_matrix = triplets
        adj1_matrix = adj1_matrix.clone().detach()
        hyperedge_index = torch.nonzero(adj1_matrix.T, as_tuple=True)
        hyperedge_index = torch.stack(hyperedge_index)
        hyperedge_index = hyperedge_index.to(x.device)
        output = self.conv1(x, hyperedge_index)
        return output

    def forward(self, X, triplet):

        output = self.process_hypergraph(triplet, X)
        return output
class Transformerblock(nn.Module):
    def __init__(
            self,
            feat_size,
            num_heads,
            bias=True,
            attn_bias_type="add",
            attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
                self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.attn_bias_type = attn_bias_type
        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.u_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.dropout = nn.Dropout(p=attn_drop)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.u_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    def forward(self, feature, attn_bias=None, attn_mask=None):
        q_h = self.q_proj(feature).transpose(0, 1)
        k_h = self.k_proj(feature).transpose(0, 1)
        v_h = self.v_proj(feature).transpose(0, 1)
        u_h = feature * torch.sigmoid(self.u_proj(feature))
        bsz, N, _ = feature.shape
        q_h = (q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)* self.scaling)
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).permute(1, 2, 0)
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn_weights = (th.bmm(q_h, k_h).transpose(0, 2).reshape(N, N, bsz, self.num_heads).transpose(0, 2))
        if attn_bias is not None:
            if self.attn_bias_type == "add":
                attn_weights += attn_bias
            else:
                attn_weights *= attn_bias
        if attn_mask is not None:
            attn_weights[attn_mask.to(th.bool)] = float("-inf")
        attn_weights = F.softmax(attn_weights.transpose(0, 2).reshape(N, N, bsz * self.num_heads).transpose(0, 2),dim=2,)
        attn_weights = self.dropout(attn_weights)
        attn = th.bmm(attn_weights, v_h).transpose(0, 1)
        attn = self.out_proj(attn.reshape(N, bsz, self.feat_size).transpose(0, 1))
        attn = u_h + attn
        return attn
class Transformerblock_add_norm(nn.Module):
    def __init__(
            self,
            feat_size,
            hidden_size,
            num_heads,
            attn_bias_type="add",
            dropout=0.1,
            attn_dropout=0.1,
    ):
        super().__init__()
        self.Transformerblock = Transformerblock(
            feat_size=feat_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=attn_dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(feat_size)
        self.norm2 = nn.LayerNorm(feat_size)
    def forward(self, feature, attn_bias=None, attn_mask=None):
        residual = feature
        feature = self.Transformerblock(feature, attn_bias, attn_mask)
        feature = self.dropout(feature)
        feature = residual + feature
        feature_norm1 = self.norm1(feature)
        feature_norm2 = self.norm2(feature_norm1)
        return feature_norm2
class confusion_of_three_channel(nn.Module):
    def __init__(self, args):
        super(confusion_of_three_channel, self).__init__()
        self.degree_encoder = DegreeEncoder(8, args.hidden)
        self.spatial_encoder = SpatialEncoder(max_dist=8, num_heads=8)
        self.AvgNeighborSim_encoder = AvgNeighborSimEncoder(args.numrna, args.numdis, args.hidden)
        self.linear = nn.Linear(2*args.hidden,args.hidden)
        self.Transformerblock_add_norm = Transformerblock_add_norm(feat_size=args.hidden,hidden_size=args.hidden,num_heads=args.numhead)
        self.bn = nn.BatchNorm1d(args.hidden)
        self.prelu = nn.PReLU()

    def forward(self, newFeature, graph,train_index,ms,ds):
        degree_embedding = self.degree_encoder(graph)
        spatial_embedding = self.spatial_encoder(graph)
        #AvgNeighbor_embedding = self.AvgNeighborSim_encoder(train_index,ms,ds)
        mix_feature = degree_embedding * degree_embedding
        # operation '*' generates the similar result to '||' with avg, and it trains faster
        mix_feature = self.bn(mix_feature)
        newFeature = newFeature + mix_feature
        newFeature = newFeature.unsqueeze(0)
        bias = spatial_embedding
        out = self.Transformerblock_add_norm(newFeature, bias)
        out = out.squeeze(0)
        return out
class final_GCN(nn.Module):
    def __init__(self, hidden_channels, num_layers, num_heads, dropout_rate=0.5):
        super(final_GCN, self).__init__()
        aggregators = ['mean', 'max', 'sum', 'symnorm']
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EGConv(hidden_channels, hidden_channels, aggregators, num_heads)
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(
                EGConv(hidden_channels, hidden_channels, aggregators, num_heads)
            )
            self.dropouts.append(nn.Dropout(dropout_rate))
        self.fc = nn.Linear(hidden_channels * (num_layers + 1), hidden_channels)
    def forward(self, data, edge_index):
        colector = [data]
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            h = conv(data, edge_index)
            h = norm(h)
            h = F.leaky_relu(h)
            h = dropout(h)
            colector.append(h)
        res_ = torch.cat(colector, dim=-1)
        res = self.fc(res_)
        return res
class MLP_for_decoding(nn.Module):
    def __init__(self, args):
        super(MLP_for_decoding, self).__init__()
        self.fc1 = nn.Linear(args.hidden, args.hidden // 2)
        self.bn1 = nn.BatchNorm1d(args.hidden // 2)
        self.dropout1 = nn.Dropout(args.MLPDropout)
        self.fc2 = nn.Linear(args.hidden // 2, args.hidden // 4)
        self.bn2 = nn.BatchNorm1d(args.hidden // 4)
        self.dropout2 = nn.Dropout(args.MLPDropout)
        self.fc3 = nn.Linear(args.hidden // 4, args.hidden // 8)
        self.bn3 = nn.BatchNorm1d(args.hidden // 8)
        self.dropout3 = nn.Dropout(args.MLPDropout)
        self.fc4 = nn.Linear(args.hidden // 8, 1)

    def forward(self, x):
        x = F.tanh(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.tanh(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.tanh(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
class Trifusion(nn.Module):
    def __init__(self, args):
        super(Trifusion, self).__init__()
        self.n_rna = args.numrna
        self.n_dis = args.numdis
        self.Linear = Linear(args)
        self.confusion_of_three_channel = confusion_of_three_channel(args)
        self.final_GCN = final_GCN(args.hidden, args.numlayer, args.numhead)
        self.hypermirna = Hyperrna(2 * args.numrna , args.hidden)
        self.hyperdis = Hyperdis(2 * args.numdis , args.hidden)
        self.MLP = MLP_for_decoding(args)
        self.Parallel_GCN_m = Parallel_GCN(args.numrna, args.hidden,args.numhead, 'mean', True, True, 0.1)
        self.Parallel_GCN_d = Parallel_GCN(args.numdis, args.hidden,args.numhead, 'mean', True, True, 0.1)
        self.bn1 = nn.BatchNorm1d(args.hidden)
        self.bn2 = nn.BatchNorm1d(args.hidden)
        self.linear_m = nn.Linear(2 * args.numrna, args.hidden)
        self.linear_d = nn.Linear(2 * args.numdis, args.hidden)
        self.linear_f = nn.Linear(2 * args.hidden, args.hidden)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def encode(self, args, similarity_feature, graph, edge_idx_dict, i, mm_graph, dd_graph, mm_matrix, dd_matrix):
        m_f = similarity_feature['m_s']['Data_M']
        d_f = similarity_feature['d_s']['Data_M']
        mm_f = similarity_feature['mm_f']['Data_M']
        mm_g = similarity_feature['mm_g']['Data_M']
        mm_s = similarity_feature['mm_s']['Data_M']
        d_ss = similarity_feature['dd_s']['Data_M']
        dd_g = similarity_feature['dd_g']['Data_M']
        m_d_graph = graph[str(i)]["fold_train_edges_80p_80n"]
        train_edge_index = edge_idx_dict[str(i)]['fold_train_edges_80p_80n']

        # first channel
        # Graph construction moved to train.py
        mf1 = self.Parallel_GCN_m(mm_graph, m_f)
        df1 = self.Parallel_GCN_d(dd_graph, d_f)
        mf2, df2 = self.Linear(args, m_f, d_f)
        mf = (mf1 + mf2) / 2
        df = (df1 + df2) / 2
        graph_fea = torch.cat([mf, df], dim=0)
        # second channel
        # rna section
        m_fea = torch.cat(((mm_f + mm_s) / 2, mm_g), dim=1)
        # use pre-calculated matrix
        m_f_sparse = mm_matrix
        m_hyper_fea1 = self.hypermirna(m_fea, m_f_sparse)
        m_hyper_fea2 = self.linear_m(m_fea)
        m_hyper_fea = (m_hyper_fea1 + m_hyper_fea2) / 2
        # dis section
        d_fea = torch.cat((d_ss, dd_g), dim=1)
        # use pre-calculated matrix
        d_f_sparse = dd_matrix
        d_hyper_fea1 = self.hyperdis(d_fea, d_f_sparse)
        d_hyper_fea2 = self.linear_d(d_fea)
        d_hyper_fea = (d_hyper_fea1 + d_hyper_fea2) / 2
        hyper_fea = torch.cat((m_hyper_fea, d_hyper_fea), dim=0)
        mix_fea = hyper_fea + graph_fea
        # third channel + fusion
        out = self.confusion_of_three_channel(mix_fea, m_d_graph, train_edge_index, m_f, d_f)
        mix_fea_residual = self.bn1(mix_fea)
        out = out + mix_fea_residual

        edge_index = to_undirected(train_edge_index)
        sumNode = self.n_rna + self.n_dis
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(sumNode, sumNode))
        result = self.final_GCN(out, adj)
        return result

    def decode(self, out, edge_label_index):
        miRNA_embedding = out[edge_label_index[0]]
        disease_embedding = out[edge_label_index[1]]
        res = (miRNA_embedding * disease_embedding)
        res = self.MLP(res)
        return res

    def forward(self, args, similarity_feature, graph, edge_idx_dict, edge_label_index, i, mm_graph, dd_graph, mm_matrix, dd_matrix):
        out = self.encode(args, similarity_feature, graph, edge_idx_dict, i, mm_graph, dd_graph, mm_matrix, dd_matrix)
        res = self.decode(out, edge_label_index)
        return res
