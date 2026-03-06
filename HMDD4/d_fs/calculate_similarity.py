
import pandas as pd
import numpy as np
from typing import Set
import math

def parse_go_terms(go_str: str) -> Set[str]:
    """解析GO terms为集合"""
    if pd.isna(go_str) or str(go_str).strip() in ['', 'nan']:
        return set()
    return set(g.strip() for g in str(go_str).split(';') if g.strip())

def penalty_based_similarity(set_i: Set[str], set_j: Set[str], alpha: float = 0.1) -> float:
    """
    改进的基于数量差异惩罚的语义相似度计算
    1. 引入 Jaccard 作为基础的全局视角特征
    2. 使用指数衰减函数 (e^(-alpha * |len_i - len_j|)) 对集合规模悬殊的对进行降权
       - alpha 控制惩罚力度，alpha 越大，对数量差异的惩罚越重
    3. 保留高置信度核心，同时抑制由于基数过大带来的假阳性关联（即“信号淹没”现象被抑制）。
    """
    len_i = len(set_i)
    len_j = len(set_j)
    
    if len_i == 0 or len_j == 0:
        return 0.0
        
    intersection_size = len(set_i & set_j)
    
    if intersection_size == 0:
        return 0.0
        
    # 基础度量：Jaccard系数 (避免了原版Overlap的极端高估现象)
    union_size = len(set_i | set_j)
    base_sim = intersection_size / union_size
    
    # 极值数量差异的非线性惩罚机制
    # 当两种疾病注释数量相差极大时，它们的总体生物学背景可能差异较大
    # 使用指数衰减进行平滑惩罚
    diff_penalty = math.exp(-alpha * abs(len_i - len_j) / max(len_i, len_j))
    
   
   
    
    final_sim = base_sim * diff_penalty
    
    return final_sim


def calculate_similarity_matrix(csv_path: str, output_path: str = None, alpha: float = 1.0):
    
    # 读取数据
    df = pd.read_csv(csv_path)
    print(f"读CSV文件: {len(df)} 行")
    
    # 解析每个疾病的GO terms
    print("解析GO terms...")
    df['go_set'] = df['go_terms'].apply(parse_go_terms)
    
    # 获取疾病名称列表
    diseases = df['disease_name'].tolist()
    n = len(diseases)
    print(f"疾病数量: {n}")
    
    # 初始化相似性矩阵
    similarity_matrix = np.zeros((n, n))
    
    # 计算相似性
    print(f"计算相似性矩阵 (指数衰减惩罚, alpha={alpha})...")
    for i in range(n):
        if i % 100 == 0:
            print(f"  进度: {i}/{n}")
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = penalty_based_similarity(df.iloc[i]['go_set'], df.iloc[j]['go_set'], alpha)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
    
    # 保存结果
    sim_df = pd.DataFrame(similarity_matrix, index=diseases, columns=diseases)
    
    if output_path is None:
        output_path = output_path
    
    sim_df.to_csv(output_path)
    print(f"\n相似性矩阵已保存至: {output_path}")
    print(f"矩阵大小: {n} x {n}")
    
    # 统计信息
    upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
    print(f"\n统计信息 (排除对角线):")
    print(f"  平均相似度: {upper_tri.mean():.6f}")
    print(f"  最大相似度: {upper_tri.max():.6f}")
    print(f"  最小相似度: {upper_tri.min():.6f}")
    print(f"  相似度为0的对数: {(upper_tri == 0).sum()}")
    
    return sim_df


if __name__ == "__main__":
    csv_path = "./disease_go.csv"
    output_path = "./d_fs.csv"
    calculate_similarity_matrix(csv_path, output_path, alpha=2.0)
