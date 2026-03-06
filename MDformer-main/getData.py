import os

import dgl
import numpy as np
import torch
from sklearn.model_selection import KFold

from utils import *


# ***************************载入图的相似性特征开始**************************
def similarity_feature_process(args):
    print("=" * 60)
    print("开始处理相似性特征数据...")
    print(f"数据路径: {args.path}")
    print(f"计算设备: {args.device}")
    
    similarity_feature = {}
    path = args.path
    device = args.device

    n_rna = np.loadtxt(os.path.join(path, 'miRNA_similarity.csv'), delimiter=',', dtype=float).shape[0]
    n_dis = np.loadtxt(os.path.join(path, 'disease_similarity.csv'), delimiter=',', dtype=float).shape[0]
    print(f"miRNA数量: {n_rna}")
    print(f"疾病数量: {n_dis}")

    print("\n1. 处理miRNA序列相似性...")
    "miRNA sequence sim"
    rna_seq_sim = np.loadtxt(os.path.join(path, '04_sequence_similarity_matrix(1).csv'), delimiter=',', dtype=float)
    print(f"   加载miRNA序列相似性矩阵: {rna_seq_sim.shape}")
    rna_seq_sim = torch.tensor(rna_seq_sim, device=device).to(torch.float32)
    rna_seq_edge_index = get_edge_index(rna_seq_sim, device)
    print(f"   生成边索引: {rna_seq_edge_index.shape}")
    g_mm_s = dgl.graph((rna_seq_edge_index[0], rna_seq_edge_index[1]))
    print(f"   创建DGL图: 节点数={g_mm_s.num_nodes()}, 边数={g_mm_s.num_edges()}")
    similarity_feature['mm_s'] = {'Data_M': rna_seq_sim, 'edges': rna_seq_edge_index, 'g': g_mm_s}

    #'Data_M'：存储相似性矩阵的PyTorch张量'edges'：存储边的索引(对于某些特征)'g'：存储DGL图对象(对于某些特征)

    print("\n2. 处理miRNA功能相似性...")
    "miRNA Gaussian sim"
    rna_Gaus_sim = np.loadtxt(os.path.join(path, 'miRNA_similarity.csv'), delimiter=',', dtype=float)
    print(f"   加载miRNA功能相似性矩阵: {rna_Gaus_sim.shape}")
    rna_Gaus_sim = torch.tensor(rna_Gaus_sim, device=device).to(torch.float32)
    rna_Gaus_edge_index = get_edge_index(rna_Gaus_sim, device)
    print(f"   生成边索引: {rna_Gaus_edge_index.shape}")
    g_mm_g = dgl.graph((rna_Gaus_edge_index[0], rna_Gaus_edge_index[1]))
    print(f"   创建DGL图: 节点数={g_mm_g.num_nodes()}, 边数={g_mm_g.num_edges()}")
    similarity_feature['mm_g'] = {'Data_M': rna_Gaus_sim, 'edges': rna_Gaus_edge_index, 'g': g_mm_g}
    
    print("\n3. 合并miRNA相似性...")
    miRNA_similarity = rna_Gaus_sim + rna_seq_sim
    print(f"   合并后的miRNA相似性矩阵形状: {miRNA_similarity.shape}")
    similarity_feature['m_s'] = {'Data_M': miRNA_similarity}
    print("   将miRNA的两种相似性矩阵相加得到综合miRNA相似性")
    print("\n4. 处理疾病语义相似性...")
    "disease semantic sim"
    dis_semantic_sim = np.loadtxt(os.path.join(path, 'd_ss_v2.csv'), delimiter=',', dtype=float)
    print(f"   加载疾病语义相似性矩阵: {dis_semantic_sim.shape}")
    dis_semantic_sim = torch.tensor(dis_semantic_sim, device=device).to(torch.float32)
    dis_sem_edge_index = get_edge_index(dis_semantic_sim, device)
    print(f"   生成边索引: {dis_sem_edge_index.shape}")
    g_dd_s = dgl.graph((dis_sem_edge_index[0], dis_sem_edge_index[1]))
    print(f"   创建DGL图: 节点数={g_dd_s.num_nodes()}, 边数={g_dd_s.num_edges()}")
    similarity_feature['dd_s'] = {'Data_M': dis_semantic_sim, 'edges': dis_sem_edge_index, 'g': g_dd_s}

    print("\n5. 处理疾病高斯相似性...")
    "disease Gaussian sim"
    dis_Gaus_sim = np.loadtxt(os.path.join(path, 'disease_similarity.csv'), delimiter=',', dtype=float)
    print(f"   加载疾病高斯相似性矩阵: {dis_Gaus_sim.shape}")
    dis_Gaus_sim = torch.tensor(dis_Gaus_sim, device=device).to(torch.float32)
    dis_Gaus_edge_index = get_edge_index(dis_Gaus_sim, device)
    print(f"   生成边索引: {dis_Gaus_edge_index.shape}")
    g_dd_g = dgl.graph((dis_Gaus_edge_index[0], dis_Gaus_edge_index[1]))
    print(f"   创建DGL图: 节点数={g_dd_g.num_nodes()}, 边数={g_dd_g.num_edges()}")
    similarity_feature['dd_g'] = {'Data_M': dis_Gaus_sim, 'edges': dis_Gaus_edge_index, 'g': g_dd_g}

    print("\n 处理疾病功能相似性")
    # dis_function_sim = np.loadtxt(os.path.join(path, 'disease_go_similarity_matrix.csv'), delimiter=',', dtype=float)
    # dis_function_sim= torch.tensor(dis_function_sim, device=device).to(torch.float32)

    print("\n6. 合并疾病相似性...")
    disease_similarity = dis_semantic_sim + dis_Gaus_sim
    print(f"   合并后的疾病相似性矩阵形状: {disease_similarity.shape}")
    print("   将疾病的两种相似性矩阵相加得到综合疾病相似性")
    similarity_feature['d_s'] = {'Data_M': disease_similarity}
    
    print("\n" + "=" * 60)
    print("相似性特征处理完成!")
    print("生成的相似性特征包括:")
    for key, value in similarity_feature.items():
        print(f"  - {key}: {list(value.keys())}")
    print("=" * 60)
    return similarity_feature


# ***************************载入图的边数据开始**************************
def load_fold_data(args):
    print("\n" + "=" * 60)
    print("开始加载图的边数据...")
    print(f"数据路径: {args.path}")
    print(f"交叉验证折数: {args.kfolds}")
    
    path = args.path
    kfolds = args.kfolds
    edge_idx_dict = dict()
    g = dict()
    
    print("\n1. 加载基础数据...")
    md_matrix = np.loadtxt(os.path.join(path + '/adj_matrix.csv'), dtype=int, delimiter=',')
    print(f"   加载miRNA-疾病关联矩阵: {md_matrix.shape}")
    print(f"   正样本数量: {np.sum(md_matrix == 1)}")
    print(f"   负样本数量: {np.sum(md_matrix == 0)}")
    edge_idx_dict['true_md'] = md_matrix
    
    m_adj = np.load(os.path.join(path, 'm_adj.npy'))
    d_adj = np.load(os.path.join(path, 'd_adj.npy'))
    print(f"   加载miRNA邻接矩阵: {m_adj.shape}")
    print(f"   加载疾病邻接矩阵: {d_adj.shape}")

    print("\n2. 准备训练和测试数据...")
    rng = np.random.default_rng(seed=42)  # 固定训练测试
    print("   使用固定随机种子42确保结果可重现")

    pos_samples = np.where(md_matrix == 1)
    pos_samples = (pos_samples[0], pos_samples[1] + md_matrix.shape[0])
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)
    print(f"   正样本索引形状: {pos_samples_shuffled.shape}")

    train_pos_edges = pos_samples_shuffled  # 11201正 90%
    print(f"   训练正样本数量: {train_pos_edges.shape[1]}")

    neg_samples = np.where(md_matrix == 0)
    neg_samples = (neg_samples[0], neg_samples[1] + md_matrix.shape[0])
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]
    print(f"   负样本索引形状: {neg_samples_shuffled.shape}")

    train_neg_edges = neg_samples_shuffled
    print(f"   训练负样本数量: {train_neg_edges.shape[1]}")

    print("\n3. 进行K折交叉验证分割...")
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    print(f"   创建{kfolds}折交叉验证器")

    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    print("   转置边索引矩阵以适配K折分割")

    train_idx, valid_idx = [], []
    for train_index, valid_index in kf.split(train_pos_edges):
        train_idx.append(train_index)
        valid_idx.append(valid_index)
    print(f"   完成K折分割，每折训练集大小: {len(train_idx[0])}, 验证集大小: {len(valid_idx[0])}")

    print(f"\n4. 为每一折生成训练和验证数据...")
    for i in range(kfolds):
        print(f"\n   处理第{i+1}折数据...")
        edges_train_pos, edges_valid_pos = train_pos_edges[train_idx[i]], train_pos_edges[valid_idx[i]]
        fold_train_pos80 = edges_train_pos.T
        fold_valid_pos20 = edges_valid_pos.T

        print(f"     训练正样本: {fold_train_pos80.shape[1]}, 验证正样本: {fold_valid_pos20.shape[1]}")

        edges_train_neg, edges_valid_neg = train_neg_edges[train_idx[i]], train_neg_edges[valid_idx[i]]
        fold_train_neg80 = edges_train_neg.T
        fold_valid_neg20 = edges_valid_neg.T

        print(f"     训练负样本: {fold_train_neg80.shape[1]}, 验证负样本: {fold_valid_neg20.shape[1]}")

        fold_100p_100n = np.hstack(
            (np.hstack((fold_train_pos80, fold_valid_pos20)), np.hstack((fold_train_neg80, fold_valid_neg20))))
        fold_train_edges_80p_80n = np.hstack((fold_train_pos80, fold_train_neg80))
        fold_train_label_80p_80n = np.hstack((np.ones(fold_train_pos80.shape[1]), np.zeros(fold_train_neg80.shape[1])))
        fold_valid_edges_20p_20n = np.hstack((fold_valid_pos20, fold_valid_neg20))
        fold_valid_label_20p_20n = np.hstack((np.ones(fold_valid_pos20.shape[1]), np.zeros(fold_valid_neg20.shape[1])))

        print(f"     合并后训练边: {fold_train_edges_80p_80n.shape[1]}, 验证边: {fold_valid_edges_20p_20n.shape[1]}")

        edge_idx_dict[str(i)] = {}
        g[str(i)] = {}
        
        print(f"     创建第{i+1}折的图数据结构...")
        # 可能用到的100
        edge_idx_dict[str(i)]["fold_100p_100n"] = torch.tensor(fold_100p_100n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_100p_100n"] = dgl.graph(
            (fold_100p_100n[0], fold_100p_100n[1])).to(device=args.device)
        print(f"       完整图(100%数据): 节点数={g[str(i)]['fold_100p_100n'].num_nodes()}, 边数={g[str(i)]['fold_100p_100n'].num_edges()}")
        
        # 训练用的80
        edge_idx_dict[str(i)]["fold_train_edges_80p_80n"] = torch.tensor(fold_train_edges_80p_80n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_train_edges_80p_80n"] = dgl.graph(
            (fold_train_edges_80p_80n[0], fold_train_edges_80p_80n[1])).to(device=args.device)
        print(f"       训练图(80%数据): 节点数={g[str(i)]['fold_train_edges_80p_80n'].num_nodes()}, 边数={g[str(i)]['fold_train_edges_80p_80n'].num_edges()}")

        edge_idx_dict[str(i)]["fold_train_label_80p_80n"] = torch.tensor(fold_train_label_80p_80n).to(torch.float32).to(
            device=args.device)
        m_adj = torch.tensor(m_adj).to(torch.int64).to(device=args.device)
        d_adj = torch.tensor(d_adj).to(torch.int64).to(device=args.device)

        edge0 = torch.tensor(fold_train_edges_80p_80n[0]).to(device=args.device)
        edge1 = torch.tensor(fold_train_edges_80p_80n[1] - md_matrix.shape[0]).to(device=args.device)
        print(f"       调整边索引: miRNA边={edge0.shape[0]}, 疾病边={edge1.shape[0]}")

        print(f"       创建异构图(训练数据)...")

        hete80 = dgl.heterograph({
            ('miRNA', 'm-d', 'disease'): (edge0, edge1),
            ('disease', 'd-m', 'miRNA'): (edge1, edge0),
            ('miRNA', 'm-m', 'miRNA'): (m_adj[0], m_adj[1]),
            ('disease', 'd-d', 'disease'): (d_adj[0], d_adj[1])
        }, device=args.device)

        print(f"       异构图节点类型: {hete80.ntypes}, 边类型: {hete80.etypes}")
        print(f"       异构图总节点数: {hete80.num_nodes()}, 总边数: {hete80.num_edges()}")

        print(f"       生成元路径随机游走(训练数据)...")

        train_mmdd_meta = []
        train_ddmm_meta = []
        train_mdmd_meta = []
        train_dmdm_meta = []
        for j in range(20):

            mmdd = dgl.sampling.random_walk(hete80, list(range(0, md_matrix.shape[0])), metapath=['m-m', 'm-d', 'd-d'])[
                0].tolist()
            for nodeList in mmdd:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            train_mmdd_meta.append(mmdd)

            ddmm = dgl.sampling.random_walk(hete80, list(range(0, md_matrix.shape[1])), metapath=['d-d', 'd-m', 'm-m'])[
                0].tolist()
            for nodeList in ddmm:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            train_ddmm_meta.append(ddmm)

            mdmd = dgl.sampling.random_walk(hete80, list(range(0, md_matrix.shape[0])), metapath=['m-d', 'd-m', 'm-d'])[
                0].tolist()
            for nodeList in mdmd:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            train_mdmd_meta.append(mdmd)

            dmdm = dgl.sampling.random_walk(hete80, list(range(0, md_matrix.shape[1])), metapath=['d-m', 'm-d', 'd-m'])[
                0].tolist()
            for nodeList in dmdm:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            train_dmdm_meta.append(dmdm)
        g[str(i)]['train_mmdd_meta'] = torch.tensor(train_mmdd_meta, requires_grad=False)
        g[str(i)]['train_ddmm_meta'] = torch.tensor(train_ddmm_meta, requires_grad=False)
        g[str(i)]['train_mdmd_meta'] = torch.tensor(train_mdmd_meta, requires_grad=False)

        g[str(i)]['train_dmdm_meta'] = torch.tensor(train_dmdm_meta, requires_grad=False)

        print(f"       训练元路径形状: mmdd={g[str(i)]['train_mmdd_meta'].shape}, ddmm={g[str(i)]['train_ddmm_meta'].shape}")


        # 验证用的20
        print(f"       创建验证图(20%数据)...")

        edge_idx_dict[str(i)]["fold_valid_edges_20p_20n"] = torch.tensor(fold_valid_edges_20p_20n).to(torch.long).to(
            device=args.device)
        g[str(i)]["fold_valid_edges_20p_20n"] = dgl.graph(
            (fold_valid_edges_20p_20n[0], fold_valid_edges_20p_20n[1])).to(device=args.device)


        print(f"       验证图: 节点数={g[str(i)]['fold_valid_edges_20p_20n'].num_nodes()}, 边数={g[str(i)]['fold_valid_edges_20p_20n'].num_edges()}")
        edge_idx_dict[str(i)]["fold_valid_label_20p_20n"] = torch.tensor(fold_valid_label_20p_20n).to(torch.float32).to(
            device=args.device)

        edge2 = torch.tensor(fold_valid_edges_20p_20n[0]).to(device=args.device)
        edge3 = torch.tensor(fold_valid_edges_20p_20n[1] - md_matrix.shape[0]).to(device=args.device)
        print(f"       调整验证边索引: miRNA边={edge2.shape[0]}, 疾病边={edge3.shape[0]}")

        print(f"       创建验证异构图...")
        hete20 = dgl.heterograph({
            ('miRNA', 'm-d', 'disease'): (edge2, edge3),
            ('disease', 'd-m', 'miRNA'): (edge3, edge2),
            ('miRNA', 'm-m', 'miRNA'): (m_adj[0], m_adj[1]),
            ('disease', 'd-d', 'disease'): (d_adj[0], d_adj[1])
        })
        print(f"       验证异构图总节点数: {hete20.num_nodes()}, 总边数: {hete20.num_edges()}")
        print(f"       生成验证元路径随机游走...")
        valid_mmdd_meta = []
        valid_ddmm_meta = []
        valid_mdmd_meta = []
        valid_dmdm_meta = []
        for jj in range(20):

            mmdd = dgl.sampling.random_walk(hete20, list(range(0, md_matrix.shape[0])),
                                            metapath=['m-m', 'm-d', 'd-d'])[0].tolist()
            for nodeList in mmdd:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            valid_mmdd_meta.append(mmdd)

            ddmm = dgl.sampling.random_walk(hete20, list(range(0, md_matrix.shape[1])),
                                            metapath=['d-d', 'd-m', 'm-m'])[0].tolist()
            for nodeList in ddmm:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            valid_ddmm_meta.append(ddmm)

            mdmd = dgl.sampling.random_walk(hete20, list(range(0, md_matrix.shape[0])),
                                            metapath=['m-d', 'd-m', 'm-d'])[0].tolist()
            for nodeList in mdmd:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            valid_mdmd_meta.append(mdmd)

            dmdm = dgl.sampling.random_walk(hete20, list(range(0, md_matrix.shape[1])),
                                            metapath=['d-m', 'm-d', 'd-m'])[0].tolist()
            for nodeList in dmdm:
                firstList = nodeList[0]
                for nodeIndex in range(len(nodeList)):
                    if nodeList[nodeIndex] == -1:
                        nodeList[nodeIndex] = firstList
            valid_dmdm_meta.append(dmdm)

        g[str(i)]['valid_mmdd_meta'] = torch.tensor(valid_mmdd_meta, requires_grad=False)
        g[str(i)]['valid_ddmm_meta'] = torch.tensor(valid_ddmm_meta, requires_grad=False)
        g[str(i)]['valid_mdmd_meta'] = torch.tensor(valid_mdmd_meta, requires_grad=False)
        g[str(i)]['valid_dmdm_meta'] = torch.tensor(valid_dmdm_meta, requires_grad=False)
        print(f"       验证元路径形状: mmdd={g[str(i)]['valid_mmdd_meta'].shape}, ddmm={g[str(i)]['valid_ddmm_meta'].shape}")
        print(f"   第{i+1}折数据处理完成!")

    print("\n" + "=" * 60)
    print("所有折的数据处理完成!")
    print(f"总共生成了{kfolds}折的训练和验证数据")
    print("每折包含:")
    print("  - 训练图(80%数据)和验证图(20%数据)")
    print("  - 异构图结构(miRNA-miRNA, disease-disease, miRNA-disease)")
    print("  - 元路径随机游走数据(mmdd, ddmm, mdmd, dmdm)")
    print("=" * 60)
    return edge_idx_dict, g
