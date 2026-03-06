

import requests
import json
import time
from typing import Optional, List, Dict, Any


def format_disease_id(cui: str, vocabulary: str = "UMLS") -> str:
   
    # 如果已经是正确格式，直接返回
    valid_prefixes = ["UMLS_", "ICD9CM_", "ICD10_", "MESH_", "OMIM_", "DO_", 
                      "EFO_", "NCI_", "HPO_", "MONDO_", "ORDO_"]
    for prefix in valid_prefixes:
        if cui.upper().startswith(prefix):
            return cui
    
    # 否则添加词汇表前缀
    return f"{vocabulary}_{cui}"


def get_genes_by_disease_cui(
    disease_cui: str,
    api_key: str,
    page_number: int = 0,
    min_score: Optional[float] = None,
    source: Optional[str] = None
) -> Dict[str, Any]:
    """
    通过疾病CUI查询DisGeNET获取关联的基因列表
    
    Args:
        disease_cui: 疾病的UMLS CUI，例如 "C0002395" 或 "UMLS_C0002395" (Alzheimer's Disease)
        api_key: DisGeNET API密钥
        page_number: 分页页码，从0开始
        min_score: 可选，最小关联分数过滤（0-1之间）
        source: 可选，数据来源过滤，如 "CURATED", "INFERRED", "ANIMAL_MODELS", "ALL"
    
    Returns:
        包含基因-疾病关联数据的字典，结构如下:
        {
            "status": str,
            "paging": {
                "totalElements": int,  # 总结果数
                "currentPageNumber": int,
                "totalElementsInPage": int
            },
            "payload": [...]  # 基因-疾病关联列表
        }
    """
    # API端点
    api_url = "https://api.disgenet.com/api/v1/gda/summary"
    
    # 格式化疾病ID（确保格式正确，如 UMLS_C0002395）
    formatted_disease_id = format_disease_id(disease_cui)
    
    # 请求参数
    params = {
        "disease": formatted_disease_id,  # 使用disease参数传入格式化后的疾病ID
        "page_number": str(page_number)
    }
    
    # 可选参数
    if min_score is not None:
        params["min_score"] = str(min_score)
    if source is not None:
        params["source"] = source
    
    # HTTP请求头
    headers = {
        "Authorization": api_key,
        "accept": "application/json"
    }
    
    # 发送请求
    response = requests.get(api_url, params=params, headers=headers, verify=False)
    
    # 处理速率限制 (429错误)
    if not response.ok:
        if response.status_code == 429:
            while not response.ok:
                wait_time = int(response.headers.get('x-rate-limit-retry-after-seconds', 60))
                print(f"达到API查询限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                print("速率限制已恢复，重新发送请求...")
                response = requests.get(api_url, params=params, headers=headers, verify=False)
                if response.ok:
                    break
        else:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
    
    return json.loads(response.text)


def get_all_genes_by_disease_cui(
    disease_cui: str,
    api_key: str,
    min_score: Optional[float] = None,
    source: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    获取疾病关联的所有基因（自动处理分页）
    
    Args:
        disease_cui: 疾病的UMLS CUI
        api_key: DisGeNET API密钥
        min_score: 最小关联分数过滤
        source: 数据来源过滤
    
    Returns:
        所有关联基因的列表
    """
    all_genes = []
    page_number = 0
    
    while True:
        result = get_genes_by_disease_cui(
            disease_cui=disease_cui,
            api_key=api_key,
            page_number=page_number,
            min_score=min_score,
            source=source
        )
        
        if "payload" in result and result["payload"]:
            all_genes.extend(result["payload"])
        
        # 检查是否还有更多页
        paging = result.get("paging", {})
        total_elements = paging.get("totalElements", 0)
        
        if len(all_genes) >= total_elements:
            break
        
        page_number += 1
        time.sleep(0.5)  # 避免请求过快
    
    return all_genes


def extract_gene_info(gda_results: List[Dict[str, Any]], top_n: int = 50) -> List[Dict[str, Any]]:
    """
    从GDA结果中提取基因信息
    
    Args:
        gda_results: 基因-疾病关联结果列表
        top_n: 最多返回的基因数量，默认50个（API已按score降序排列）
    
    Returns:
        提取的基因信息列表，每个包含:
        - gene_symbol: 基因符号
        - gene_ncbi_id: NCBI基因ID  
        - score: DisGeNET关联分数
        - evidence_count: 证据数量
    """
    genes = []
    seen_genes = set()
    
    for gda in gda_results:
        gene_symbol = gda.get("symbolOfGene") or gda.get("geneSymbol") or gda.get("gene_symbol", "")
        gene_ncbi_id = gda.get("geneNcbiID") or gda.get("gene_ncbi_id", "")
        
        # 去重
        if gene_ncbi_id in seen_genes:
            continue
        seen_genes.add(gene_ncbi_id)
        
        genes.append({
            "gene_symbol": gene_symbol,
            "gene_ncbi_id": gene_ncbi_id,
            "score": gda.get("score", 0),
            "evidence_count": gda.get("evidenceCount") or gda.get("evidence_count", 0)
        })
        
        # 达到top_n限制则停止
        if len(genes) >= top_n:
            break
    
    return genes


if __name__ == "__main__":
    import pandas as pd
    import time
    
    API_KEY = "2bfbbdda-e1c9-42e9-b2b9-19f4f418e1c4"
    
    # 读取CSV文件
    csv_path = r"d:\MiRnaProject\MDformer-main\HMDD4\semantic\functional\disease_gene_thired.csv"
    df = pd.read_csv(csv_path)
    
    print(f"读取CSV文件: {len(df)} 行")
    print(f"列名: {list(df.columns)}")
    
    # 统计需要处理的行数（有CUI但genes列为空或基因数为0的行）
    needs_query = df[(df['cui'].notna()) & (df['cui'].str.strip() != '')]
    print(f"需要查询基因的行数: {len(needs_query)}")
    
    # 遍历每一行，查询基因
    updated_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        cui = row.get('cui', '')
        genes_existing = row.get('genes', '')
        gene_count = row.get('gene_count', 0)
        
        # 跳过没有CUI的行
        if pd.isna(cui) or str(cui).strip() == '':
            continue
        
        # 跳过已经有基因数据的行
        if not pd.isna(genes_existing) and str(genes_existing).strip() != '' and gene_count > 0:
            continue
        
        cui = str(cui).strip()
        disease_name = row.get('disease_name', 'Unknown')
        
        print(f"\n[{idx+1}/{len(df)}] 查询: {disease_name} (CUI: {cui})")
        
        try:
            # 查询基因
            result = get_genes_by_disease_cui(
                disease_cui=cui,
                api_key=API_KEY,
                page_number=0
            )
            
            if result.get('status') == 'OK' and 'payload' in result:
                # 提取前50个基因
                genes = extract_gene_info(result['payload'], top_n=50)
                gene_symbols = [g['gene_symbol'] for g in genes if g['gene_symbol']]
                
                # 以分号分隔写入
                genes_str = ';'.join(gene_symbols)
                df.at[idx, 'genes'] = genes_str
                df.at[idx, 'gene_count'] = len(gene_symbols)
                
                print(f"  找到 {len(gene_symbols)} 个基因: {genes_str[:80]}...")
                updated_count += 1
            else:
                print(f"  无结果或错误: {result.get('status', 'unknown')}")
                error_count += 1
            
            # 避免API限流，每次请求后等待
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  错误: {e}")
            error_count += 1
            time.sleep(1)  # 出错后等待更长时间
    
    # 保存结果
    output_path = 'scrape_with_genes_final.csv'
    df.to_csv(output_path, index=False)
    print(f"\n完成! 更新了 {updated_count} 行, 错误 {error_count} 行")
    print(f"结果保存至: {output_path}")
