import datetime

from getData import *
from model import *


def fold_valid(args):
    print("\n📁 步骤1: 加载和处理数据...")
    similarity_feature = similarity_feature_process(args)
    edge_idx_dict, g = load_fold_data(args)
    n_rna = edge_idx_dict['true_md'].shape[0]
    n_dis = edge_idx_dict['true_md'].shape[1]
    print(f"✅ 数据加载完成 - miRNA: {n_rna}, 疾病: {n_dis}")

    print("\n🏗️  步骤2: 初始化模型...")
    model = MY_Module(args, n_rna, n_dis).to(args.device)
    print("✅ 模型初始化完成")
    print("*******************************************************************")
    metric_result_list = []
    metric_result_list_str = []
    metric_result_list_str.append('AUC    AUPR    Acc    F1    pre    recall')
    print(f"\n🔄 步骤3: 开始{args.kfolds}折交叉验证训练...")
    for i in range(args.kfolds):
        model = MY_Module(args, n_rna, n_dis).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

        print(f'\n📊 第{i + 1}折训练开始...')
        Record_res = []
        Record_res.append('AUC    AUPR    Acc    F1    pre    recall')
        model.train()
        for epoch in range(args.epoch):
            optimizer.zero_grad()

            out = model(args, similarity_feature, g, edge_idx_dict, edge_idx_dict[str(i)]['fold_train_edges_80p_80n'],
                        i).view(-1)

            loss = criterion(out, edge_idx_dict[str(i)]['fold_train_label_80p_80n'])
            loss.backward()
            optimizer.step()

            test_auc, metric_result, y_true, y_score = valid_fold(args, model,
                                                                  similarity_feature,
                                                                  g,
                                                                  edge_idx_dict,
                                                                  edge_idx_dict[str(i)]['fold_valid_edges_20p_20n'],
                                                                  i
                                                                  )
            One_epoch_metric = '{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*metric_result)
            Record_res.append(One_epoch_metric)
            if epoch + 1 == args.epoch:
                metric_result_list.append(metric_result)
                metric_result_list_str.append(One_epoch_metric)
            print('  Epoch {:03d} | Loss: {:.6f} | Val_AUC: {:.4f}'.format(epoch + 1, loss.item(), test_auc))
        
        print(f'✅ 第{i + 1}折训练完成')

    print(f"\n📈 步骤4: 计算最终结果...")
    arr = np.array(metric_result_list)
    averages = np.round(np.mean(arr, axis=0), 4)
    metric_result_list_str.append('平均值：')
    metric_result_list_str.append('{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*list(averages)))

    print(f"💾 保存结果到文件...")
    now = datetime.datetime.now()
    filename = '平均_' + now.strftime("%Y_%m_%d_%H_%M_%S") + '_.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(metric_result_list_str))
    print(f"✅ 结果已保存到: {filename}")
    return averages


def valid_fold(args, model, similarity_feature, graph, edge_idx_dict, edge_label_index, i):
    lable = edge_idx_dict[str(i)]['fold_valid_label_20p_20n']

    model.eval()
    with torch.no_grad():
        out = model.encode(args, similarity_feature, graph, edge_idx_dict, i)
        res = model.decode(out, edge_label_index, i).view(-1).sigmoid()
        model.train()
    metric_result = caculate_metrics(lable.cpu().numpy(), res.cpu().numpy())
    my_acu = metrics.roc_auc_score(lable.cpu().numpy(), res.cpu().numpy())
    return my_acu, metric_result, lable, res
