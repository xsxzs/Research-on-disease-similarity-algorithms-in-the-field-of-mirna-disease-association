import csv
import os
import pandas as pd

# =================配置=================
# 输入文件 (包含 disease_name 和 mesh_ids)
INPUT_CSV = '../test_v2.csv'
# 基因库文件
DB_FILE = 'DisGeNET.txt'
# 输出文件
OUTPUT_CSV = 'disease_genes_final.csv'

def load_enrichr_db(file_path):
    """
    加载 Enrichr 数据库，构建 {小写名字: 基因列表} 的字典
    """
    db = {}
    print(f"正在加载基因库: {file_path} ...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 分割行：Enrichr 格式通常是 Tab 分隔
            # [0] = 疾病名, [1] = 描述(可能为空), [2:] = 基因
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            
            if len(parts) < 2:
                continue
            
            raw_name = parts[0]
            # 提取基因：排除第一列后的所有大写字符串通常是基因
            # 这里简单处理：取 parts[1:] 作为基因列表
            # 因为 Enrichr 格式里，第二列有时是空描述，有时直接是基因，
            # 但 parts 列表是通过 split('\t') 并过滤空串得到的，
            # 所以只要不是第一列，通常都是基因。
            genes = parts[1:]
            
            # 核心逻辑：统一转小写，作为匹配的 Key
            # 这样 'Renal Fibrosis' 和 'renal fibrosis' 就能匹配上了
            key = raw_name.lower()
            
            # 如果有重复名字，保留基因更多的那条（或者合并，这里先覆盖）
            if key not in db or len(genes) > len(db[key]):
                db[key] = genes
                
    print(f"数据库加载完毕，索引了 {len(db)} 个唯一疾病名。")
    return db

def main():
    # 1. 检查输入
    if not os.path.exists(INPUT_CSV):
        print(f"错误: 找不到输入文件 {INPUT_CSV}")
        return
    if not os.path.exists(DB_FILE):
        print(f"错误: 找不到数据库文件 {DB_FILE}")
        return

    # 2. 加载数据库
    gene_db = load_enrichr_db(DB_FILE)
    
    # 3. 读取输入 CSV
    # 保持原始顺序，不做任何排序
    df = pd.read_csv(INPUT_CSV)
    print(f"读取输入列表: {len(df)} 行")
    
    # 4. 准备新列数据
    gene_counts = []
    gene_lists = []
    
    match_success = 0
    
    # 5. 逐行匹配
    for index, row in df.iterrows():
        # 获取疾病名 (兼容不同列名写法)
        dname = str(row.get('disease_name', row[0])).strip()
        
        # 匹配逻辑：转小写
        target_key = dname.lower()
        
        if target_key in gene_db:
            # 命中！
            genes = gene_db[target_key]
            gene_counts.append(len(genes))
            gene_lists.append(';'.join(genes))
            match_success += 1
        else:
            # 未命中
            # 这里是严格匹配，没有做去数字或模糊匹配
            gene_counts.append(0)
            gene_lists.append('') # 空
            
    # 6. 添加新列
    df['gene_count'] = gene_counts
    df['genes'] = gene_lists
    
    # 7. 保存结果
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    # 8. 报告
    print("="*40)
    print(f"处理完成。")
    print(f"输入总数: {len(df)}")
    print(f"成功匹配: {match_success} ({match_success/len(df):.2%})")
    print(f"未匹配: {len(df) - match_success}")
    print(f"结果已保存至: {OUTPUT_CSV}")
    print("="*40)

if __name__ == "__main__":
    main()
