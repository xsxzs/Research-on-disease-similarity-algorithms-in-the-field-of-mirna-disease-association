import os
from itertools import count

import pandas as pd
import pandas as pd
import os

# 1. 读取主库匹配结果 (Descriptor matches - D-series IDs)
# matchmesh.csv 应该是你第一次通过 desc2025.xml 匹配出来的初步结果
main_df = pd.read_csv('./matchmesh.csv', encoding='utf-8')

# 2. 读取补充库匹配结果 (Supplemental matches - C-series IDs)
# fixed_matches1.csv 是通过 missing_disease.py 从 supp2025.xml 中找回来的 ID
supp_df = pd.read_csv('./fixed_matches1.csv', encoding='utf-8')

print("开始合并主库(desc)与补充库(supp)的匹配结果...")

# 3. 核心合并逻辑：使用补充库的 C ID 覆盖/填补主库中缺失或不准的 ID
# 遍历补全集中的每一行
for i, row in supp_df.iterrows():
    disease_name = row['disease_name']
    mesh_id = str(row['mesh_ids']).strip()
    
    # 只有当补全集中有有效的 ID 时才进行操作
    if pd.notna(row['mesh_ids']) and mesh_id != '':
        # 在主结果中找到相同疾病名的行，并更新其 mesh_ids
        main_df.loc[main_df['disease_name'] == disease_name, 'mesh_ids'] = mesh_id

# 4. 统计合并后的缺失情况
missing_count = main_df['mesh_ids'].isna().sum()
print(f"合并完成。当前仍有 {missing_count} 个疾病未匹配到 MeSH ID。")

# 5. 保存最终版本
# 保存为 disease_desc.csv，供后续相似度计算脚本使用
output_file = './final_data.csv'
main_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"最终结果已保存至: {output_file}")