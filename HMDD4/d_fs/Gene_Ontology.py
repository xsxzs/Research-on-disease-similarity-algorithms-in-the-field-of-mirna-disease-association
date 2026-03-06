"""
使用g:Profiler API获取基因GO注释，并计算疾病功能相似性
API文档: https://biit.cs.ut.ee/gprofiler/page/apis
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Set
import time


def get_go_annotations(genes: List[str], organism: str = 'hsapiens') -> Dict[str, Set[str]]:
    """
    使用g:Profiler API获取基因列表的GO注释
    
    注意: 这里使用gost/profile端点，设置宽松阈值来获取GO terms
    
    Args:
        genes: 基因符号列表，如 ["CASQ2", "DMD", "GSTM1"]
        organism: 物种，默认人类 'hsapiens'
    
    Returns:
        字典，键为基因符号，值为该基因关联的GO term ID集合
    """
    if not genes:
        return {}
    
    # 调用g:Profiler API
    r = requests.post(
        url='https://biit.cs.ut.ee/gprofiler/api/gost/profile/',
        json={
            'organism': organism,
            'query': genes,
            'sources': ['GO:BP', 'GO:MF', 'GO:CC'],  # 三种GO类别
            'user_threshold': 0.1,  # 只返回p < 0.1的GO terms
            'all_results': False,  # 只返回显著结果
            'no_evidences': True,  
            'no_iea': False, 
        },
        headers={
            'User-Agent': 'DiseaseSimilarityCalculator'
        }
    )
    
    result = r.json()
    
    # 收集所有GO terms
    go_terms = set()
    if 'result' in result and result['result']:
        for item in result['result']:
            native_id = item.get('native', '')
            if native_id.startswith('GO:'):
                go_terms.add(native_id)
    
    # 返回整个基因列表对应的GO terms（作为集合）
    return go_terms


def get_disease_go_terms(genes_str: str) -> Set[str]:
    """
    获取一个疾病（通过其基因列表）的所有GO terms
    
    Args:
        genes_str: 分号分隔的基因字符串，如 "CASQ2;DMD;GSTM1"
    
    Returns:
        该疾病关联的GO term ID集合
    """
    if pd.isna(genes_str) or not genes_str.strip():
        return set()
    
    genes = [g.strip() for g in str(genes_str).split(';') if g.strip()]
    if not genes:
        return set()
    
    return get_go_annotations(genes)



if __name__ == "__main__":

    csv_path = "./scrape_with_genes_final.csv"
    df = pd.read_csv(csv_path)
    
    print(f"读取CSV文件: {len(df)} 行")
    
    # 筛选有基因数据的疾病
    df_with_genes = df[df['genes'].notna() & (df['genes'] != '') & (df['go_count'] <4 ) ].copy()
    print(f"有基因数据的疾病: {len(df_with_genes)} 个")
    
    # 存储GO注释
    df['go_terms'] = ''
    df['go_count'] = 0
    
 
    for idx, row in df_with_genes.iterrows():
        disease_name = row['disease_name']
        genes_str = row['genes']
        
        print(f"\n[{idx+1}] {disease_name[:40]}...")
        
        try:
            go_terms = get_disease_go_terms(genes_str)
            go_terms_str = ';'.join(sorted(go_terms))
            
            df.at[idx, 'go_terms'] = go_terms_str
            df.at[idx, 'go_count'] = len(go_terms)
            
            print(f"  基因数: {row['gene_count']}, GO terms数: {len(go_terms)}")
            
            # time.sleep(0.5)
            
        except Exception as e:
            print(f"  错误: {e}")
            time.sleep(1)
    
   
    import csv
    output_path = './disease_go.csv'
    df.to_csv(output_path, index=False)
    print(f"\n完成! 结果保存至: {output_path}")
