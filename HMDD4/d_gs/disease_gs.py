import pandas as pd
import numpy as np


def calculate_gip_similarity(interaction_matrix, entity_type="Entity"):

    matrix = interaction_matrix.values  # 获取交互矩阵的数值部分

    # 计算矩阵的点积
    dot_product = np.dot(matrix, matrix.T)
    # 计算矩阵行向量的平方范数
    sq_norms = np.diag(dot_product)
    # 计算平方欧氏距离
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * dot_product
    # 确保距离值非负
    dist_sq = np.maximum(dist_sq, 0)

    # 计算 Gamma 参数
    # 计算每行元素的平方和
    row_norms_sq = np.sum(matrix ** 2, axis=1)
    # 计算平均平方范数
    mean_norm_sq = np.mean(row_norms_sq)

    # 处理平均平方范数为0的特殊情况
    if mean_norm_sq == 0:
        gamma = 1
    else:
        gamma = 1.0 / mean_norm_sq  # 计算Gamma参数

    print(f"  - {entity_type} Gamma 参数: {gamma:.5f}")

    # 计算核矩阵（高斯核）
    gip_matrix = np.exp(-gamma * dist_sq)

    # 将结果转换为DataFrame，保留原始索引
    return pd.DataFrame(gip_matrix, index=interaction_matrix.index, columns=interaction_matrix.index)


def process_hmdd_ordered(file_path):

  
    # read CSV
    df = pd.read_csv(file_path)  # 使用pandas读取CSV文件，数据存储在df变量中


    # 使用原始列的 unique()，保留出现顺序
    ordered_diseases = df['disease'].unique()  # 获取并保存所有疾病名称，保持原始出现顺序

    # 统一小写，然后取 unique()，保留出现顺序

    # 对miRNA名称进行清洗：转换为小写并去除首尾空格
    df['clean_miRNA'] = df['miRNA'].str.lower().str.strip()  # 创建新列，存储清洗后的miRNA名称（小写且无空格）
    ordered_mirnas = df['clean_miRNA'].unique()  # 获取并保存所有清洗后的miRNA名称，保持原始出现顺序


    # 使用交叉表(crosstab)构建初始关联矩阵
    # crosstab 默认会按字母排序
    temp_matrix = pd.crosstab(df['clean_miRNA'], df['disease'])  # 创建miRNA与疾病的交叉表，形成初始关联矩阵


    # 使用 reindex 强制按照 ordered_mirnas 和 ordered_diseases 排序

    # 并将缺失值填充为0
    adj_matrix = temp_matrix.reindex(index=ordered_mirnas, columns=ordered_diseases, fill_value=0)  # 按照指定顺序重新索引矩阵，缺失值填充为0

    #二值化处理：将大于0的值设为1，其余为0
    adj_matrix = (adj_matrix > 0).astype(int)  # 将关联矩阵二值化，1表示存在关联，0表示无关联



    # 打印矩阵信息用于调试和核对
    print(f"最终矩阵维度: {adj_matrix.shape}")  # 打印最终矩阵的维度（行数×列数）
    print(f"  - 第一行 miRNA: {adj_matrix.index[0]}")  # 打印第一个miRNA名称
    print(f"  - 第一列 疾病: {adj_matrix.columns[0]}")  # 打印第一个疾病名称



    # 保存关联矩阵到CSV文件，不包含列名和索引
    adj_matrix.to_csv('adj_matrix.csv', header=False, index=False)  # 将关联矩阵保存为CSV文件

    # miRNA GIP
    mirna_gip = calculate_gip_similarity(adj_matrix, "miRNA")

    mirna_gip.to_csv('miRNA_similarity.csv', header=False, index=False)

    # 疾病 GIP
    disease_gip = calculate_gip_similarity(adj_matrix.T, "Disease")

    disease_gip.to_csv('disease_similarity.csv', header=False, index=False)

    # 保存索引用于核对

    with open('names_miRNAs.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(adj_matrix.index.astype(str)))

    with open('names_diseases.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(adj_matrix.columns.astype(str)))


if __name__ == "__main__":
    file_path = 'data.csv'
    try:
        process_hmdd_ordered(file_path)

    except FileNotFoundError:
        print(f"找不到文件 {file_path}")