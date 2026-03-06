import subprocess
import sys
import time


scripts_to_run = [
    "match_genes_with_DisGeNET.py",          # 基础名称精准匹配基因
    "match_genes_with_synonyms.py",  #  同义词匹配抢救
    "selenium_scrape_cui_by_diseasename.py",        # 爬虫：根据 疾病名称 找 CUI
    "scrape_gene.py",                #  API：根据 CUI 获取关联基因
    "Gene_Ontology.py",              #  API：根据 基因获取 GO 注释
    "calculate_similarity.py"        #  核心：计算相似度矩阵
]

def print_separator(title):
    print("\n" + "=" * 60)
    print(f">>> 开始执行阶段: {title} <<<")
    print("=" * 60 + "\n")

print("🚀 开始全流程自动化数据处理 Pipeline 🚀\n")
total_start_time = time.time()

for script in scripts_to_run:
    print_separator(script)
    step_start_time = time.time()
    
    try:
        # 使用 subprocess.run 执行脚本
      
        result = subprocess.run([sys.executable, script], check=True)
        
        step_cost = time.time() - step_start_time
        print(f"\n✅ [成功] {script} 执行完毕，耗时: {step_cost:.2f} 秒\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [致命错误] 脚本 {script} 执行失败 (返回码: {e.returncode})！")
        sys.exit(1)
    except FileNotFoundError:
         print(f"\n❌ [找不到文件] 无法在当前目录找到脚本: {script}\n")
         sys.exit(1)

total_cost = time.time() - total_start_time
print("=" * 60)
print(f"🎉 所有 {len(scripts_to_run)} 个步骤执行完毕！全流程总耗时: {total_cost:.2f} 秒")
print("=" * 60)
