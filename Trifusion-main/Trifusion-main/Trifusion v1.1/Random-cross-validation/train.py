from get_Data import *
from model import *
import warnings
from param import *
import networkx as nx
import dgl
from utils import k_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
def train(args):
    similarity_feature = similarity_feature_process(args)
    edge_idx_dict, g = load_fold_data(args)

    print("*************************starting the train*****************************")
    print("***********************Iterate every 10 seconds *****************************")
    
    # Pre-calculate graphs and matrices here to avoid doing it in every epoch
    print("Pre-calculating graphs and matrices...")
    mm_matrix = k_matrix(similarity_feature['m_s']['Data_M'], args.numneighbor, args.device)
    dd_matrix = k_matrix(similarity_feature['d_s']['Data_M'], args.numneighbor, args.device)
    
    # Convert to tensors for passing to model
    mm_matrix_tensor = torch.tensor(mm_matrix, device=args.device).float()
    dd_matrix_tensor = torch.tensor(dd_matrix, device=args.device).float()

    # Optimized graph construction avoiding NetworkX bottleneck
    # Find non-zero indices directly from the tensor
    mm_src, mm_dst = torch.nonzero(mm_matrix_tensor, as_tuple=True)
    dd_src, dd_dst = torch.nonzero(dd_matrix_tensor, as_tuple=True)
    
    # Create DGL graphs from edge indices
    mm_graph = dgl.graph((mm_src, mm_dst), num_nodes=mm_matrix.shape[0]).to(args.device)
    dd_graph = dgl.graph((dd_src, dd_dst), num_nodes=dd_matrix.shape[0]).to(args.device)
    
    print("Graph calculation finished.")

    metric_result_list = []
    metric_result_list_str = []
    metric_result_list_str.append('AUC          AUPR         Acc         F1        pre          recall')
    all_loss_history = {}  # 记录每折的 loss 收敛曲线
    for i in range(args.kfolds):
        model = Trifusion(args).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

        print(f'###########################Fold {i + 1} of {args.kfolds}###########################')
        Record_res = []
        Record_res.append('AUC        AUPR      Acc       F1      pre        recall')
        model.train()
        loss_history = []  # 记录当前折的 loss
        for epoch in range(args.epoch):
            optimizer.zero_grad()

            out = model(args, similarity_feature, g, edge_idx_dict, edge_idx_dict[str(i)]['fold_train_edges_80p_80n'],
                        i, mm_graph, dd_graph, mm_matrix_tensor, dd_matrix_tensor).view(-1)

            loss = criterion(out, edge_idx_dict[str(i)]['fold_train_label_80p_80n'])
            loss.backward()
            optimizer.step()

            test_auc, metric_result, y_true, y_score = valid(args, model,similarity_feature,g,edge_idx_dict,edge_idx_dict[str(i)]['fold_valid_edges_20p_20n'],i, mm_graph, dd_graph, mm_matrix_tensor, dd_matrix_tensor
                                                                  )
            One_epoch_metric = '{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*metric_result)
            Record_res.append(One_epoch_metric)
            if epoch + 1 == args.epoch:
                metric_result_list.append(metric_result)
                metric_result_list_str.append(One_epoch_metric)
            loss_val = loss.item()
            loss_history.append(loss_val)
            print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} '.format(epoch + 1, loss_val, test_auc))
        all_loss_history[i + 1] = loss_history

    arr = np.array(metric_result_list)
    averages = np.round(np.mean(arr, axis=0), 4)
    metric_result_list_str.append('average:')
    metric_result_list_str.append('{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*list(averages)))


    with open('result ' +'_'+ str(averages[0]) +'_.txt', 'w') as f:
        f.write('\n'.join(metric_result_list_str))

    # ==================== 绘制 Loss 收敛曲线 ====================
    save_dir = os.path.dirname(os.path.abspath(__file__))
    plt.figure(figsize=(10, 6))
    for fold_id, losses in all_loss_history.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f'Fold {fold_id}', alpha=0.8)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('BCEWithLogits Loss', fontsize=14)
    plt.title('Training Loss Convergence Curve', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_fig_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()
    print(f"Loss 收敛曲线已保存: {loss_fig_path}")

    # ==================== 保存预测矩阵 ====================
    print("*************************生成预测矩阵*****************************")
    model.eval()
    with torch.no_grad():
        # 使用最后一折的模型生成所有节点的 embedding
        last_fold = args.kfolds - 1
        out = model.encode(args, similarity_feature, g, edge_idx_dict, last_fold,
                           mm_graph, dd_graph, mm_matrix_tensor, dd_matrix_tensor)

        # 构造所有 miRNA-disease 对的 edge_index
        # miRNA 节点索引: 0 ~ numrna-1
        # disease 节点索引: numrna ~ numrna+numdis-1
        mirna_indices = torch.arange(args.numrna, device=args.device)
        disease_indices = torch.arange(args.numrna, args.numrna + args.numdis, device=args.device)

        # 生成所有 miRNA-disease 对 (numrna * numdis 个)
        mirna_repeat = mirna_indices.repeat_interleave(args.numdis)  # 每个miRNA重复numdis次
        disease_repeat = disease_indices.repeat(args.numrna)          # 所有disease重复numrna次
        all_pairs = torch.stack([mirna_repeat, disease_repeat], dim=0)

        # 分批 decode 以避免显存不足
        batch_size = 100000
        num_pairs = all_pairs.shape[1]
        all_scores = []
        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            batch_pairs = all_pairs[:, start:end]
            batch_scores = model.decode(out, batch_pairs).view(-1).sigmoid()
            all_scores.append(batch_scores.cpu())

        all_scores = torch.cat(all_scores, dim=0).numpy()

        # reshape 为 (numrna, numdis) 矩阵
        prediction_matrix = all_scores.reshape(args.numrna, args.numdis)

        # 保存
        # np.save(os.path.join(save_dir, 'prediction_matrix.npy'), prediction_matrix)
        # np.savetxt(os.path.join(save_dir, 'prediction_matrix.csv'), prediction_matrix, delimiter=',', fmt='%.6f')
        # print(f"预测矩阵已保存! 维度: {prediction_matrix.shape}")
        # print(f"  - prediction_matrix.npy")
        # print(f"  - prediction_matrix.csv")
    # ==================== 预测矩阵保存完成 ====================

    return averages


def valid(args, model, similarity_feature, g, edge_idx_dict, edge_label_index, i, mm_graph, dd_graph, mm_matrix, dd_matrix):
    lable = edge_idx_dict[str(i)]['fold_valid_label_20p_20n']

    model.eval()
    with torch.no_grad():
        out = model.encode(args, similarity_feature, g, edge_idx_dict, i, mm_graph, dd_graph, mm_matrix, dd_matrix)
        res = model.decode(out, edge_label_index).view(-1).sigmoid()
        model.train()
    metric_result = caculate_metrics(lable.cpu().numpy(), res.cpu().numpy())
    my_acu = metrics.roc_auc_score(lable.cpu().numpy(), res.cpu().numpy())
    return my_acu, metric_result, lable, res

def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    average_result = train(args)
    print(average_result)
    print("finish")

if __name__ == '__main__':
    main()