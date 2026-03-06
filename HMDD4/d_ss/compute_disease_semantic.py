import os
import csv
import math
import numpy as np
import pandas as pd
import obonet
import networkx as nx
import xml.etree.ElementTree as ET

#MeSH方法采用的是基于语义贡献传播和信息内容（IC）加权的混合方法
def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)





def parse_mesh_xml(xml_path,supp_path):
    """
    解析MeSH XML文件和补充记录文件，提取疾病信息并构建DAG边

    参数:
        xml_path: MeSH主XML文件路径
        supp_path: MeSH补充记录XML文件路径

    返回:
        desc_list: 描述符列表，每个描述符包含ui、name、terms和trees
        edges_set: DAG边集合，每个边表示父子关系
    """
    # 解析主XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    desc_list = []
    # 遍历所有描述符记录
    for desc in root.findall('.//DescriptorRecord'):
        # 提取描述符唯一标识符(UI)
        ui_el = desc.find('./DescriptorUI')
        # 提取描述符名称
        name_el = desc.find('./DescriptorName/String')

        # 跳过不完整的记录
        if ui_el is None or name_el is None:
            continue

        ui = ui_el.text.strip()
        pname = name_el.text.strip()

        # 提取同义词/入口术语
        entry_terms = []
        for et in desc.findall('./ConceptList/Concept/TermList/Term/String'):
            if et.text:
                entry_terms.append(et.text.strip())

        # 提取树号，用于构建层次结构
        tree_numbers = []
        for tn in desc.findall('./TreeNumberList/TreeNumber'):
            if tn.text:
                tree_numbers.append(tn.text.strip())

        # 将描述符信息添加到列表
        desc_list.append({
            'ui': ui,              # 唯一标识符
            'name': pname,         # 主名称
            'terms': entry_terms,  # 同义词列表
            'trees': tree_numbers  # 树号列表
        })

    # 构建树号到UI的映射关系
    tn_to_ui = {}
    for d in desc_list:
        for tn in d['trees']:
            tn_to_ui[tn] = d['ui']

    # 基于树号构建DAG边（父子关系）
    edges_set = set()
    for tn, ui in tn_to_ui.items():
        if '.' in tn:
            # 例如 "C01.1.2" 的父节点是 "C01.1"
            parent_tn = tn.rsplit('.', 1)[0]
            # 查找父节点的UI
            p_ui = tn_to_ui.get(parent_tn)
            if p_ui:
                # 添加边：子节点 -> 父节点
                edges_set.add((ui, p_ui))

    # 解析补充记录文件，添加额外的边关系
    count_add = 0
    tree1 = ET.parse(supp_path)
    root1 = tree1.getroot()

    # 遍历所有补充记录
    for record in root1.findall('.//SupplementalRecord'):
        # 提取补充记录的UI
        ui_el = record.find('./SupplementalRecordUI')
        if ui_el is None or ui_el.text is None:
            continue
        child_ui = ui_el.text.strip()

        # 查找映射到主描述符的父节点
        d_parents = record.findall('.//HeadingMappedTo/DescriptorReferredTo/DescriptorUI')
        for p_el in d_parents:
            if p_el.text:
                # 清理UI中的特殊字符
                clean_id = p_el.text.replace('*', '').strip()
                # 添加边：补充记录 -> 主描述符
                edges_set.add((child_ui, clean_id))
                count_add += 1

        # 查找映射到其他补充记录的父节点
        c_parents = record.findall('.//HeadingMappedTo/SupplementalRecordReferredTo/SupplementalRecordUI')
        for p_el in c_parents:
            if p_el.text:
                # 清理UI中的特殊字符
                clean_id = p_el.text.replace('*', '').strip()
                # 添加边：补充记录 -> 补充记录
                edges_set.add((child_ui, clean_id))
                count_add += 1

    # 输出从补充记录中添加的边数统计
    print("最后添加的边数：", count_add)

    return desc_list, edges_set


def compute_mesh_similarity(name_to_mesh, edges, alpha=0.5):
    """

    
    该方法结合了Wang-style语义贡献和IC加权的思想
    
    1. 构建MeSH DAG（有向无环图）：
       - 子节点指向父节点（从更具体到更抽象）
       
    2. 计算语义贡献 S_d(t)：
       对每个疾病d，计算它对其祖先术语t的语义贡献值
       - 使用宽度优先搜索从疾病的MeSH术语向上传播
       - 贡献值 = alpha^步数，步数表示从疾病术语到祖先的层次距离
       - alpha通常为0.5，表示每上一层贡献减半
       
       S_d(t) = Σ(alpha^(从每个d的MeSH术语到t的步数))
       
    3. 计算信息内容 IC(t)：
       - IC(term) = -log(P(term))
       - P(term) = (包含该term的疾病数量 + 1) / 总疾病数量
       
    4. DSS1（未加权的语义相似度）：
       - DSS1(i,j) = Σ(S_i(t) + S_j(t))共享词条 / Σ(S_i(t) + S_j(t))所有词条
       - 计算共享语义贡献的总和与总语义贡献的比值
       
    5. DSS2（IC加权的语义相似度）：
       - DSS2(i,j) = Σ((S_i(t) + S_j(t)) * IC(t))共享 / Σ((S_i(t) + S_j(t)) * IC(t))所有
       - 在DSS1的基础上，用IC值对每个术语加权
       - IC值高的术语（更稀有）贡献更大
       
    6. 最终相似度：
       - S = (DSS1 + DSS2) / 2
       
    例子：
       疾病A: 甲状腺癌 (C04.1)
       疾病B: 癌症 (C01)
       
       对于术语"C04"（甲状腺疾病）：
       - A的贡献：alpha^1（一级父节点）
       - B的贡献：alpha^2（两级父节点）
       
       如果两者共享"C04.1"到"C04"的路径，DSS基于共享的部分计算
    """
    #  构建DAG
    g = nx.DiGraph()
    g.add_edges_from(edges)
    
    # 计算语义贡献
    def semantic_contrib_from_concept(concept_id):
        """
        计算指定 MeSH 概念及其所有祖先节点的语义贡献值。
        
        算法逻辑：
        1. 初始节点的贡献值为 1.0。
        2. 采用 BFS/DFS 策略沿着 DAG 边向父节点（更抽象的概念）传播。
        3. 传播规则：父节点的贡献值 = 当前节点贡献值 * alpha (通常为 0.5)。
        4. 如果一个节点可以通过多条路径到达，则保留贡献值最大的那条路径。
        
        参数:
            concept_id: 起始 MeSH ID (Descriptor UI 或 Supplemental UI)
        返回:
            dict: 键为 MeSH ID，值为该 ID 对起始概念的语义贡献贡献值
        """
        contrib = {concept_id: 1.0}  # 自身贡献为1
        frontier = [(concept_id, 1.0)]  # (节点, 当前贡献值)
        visited = {concept_id}
        
        while frontier:
            nid, val = frontier.pop()
            # 向所有父节点传播，贡献值乘以0.5
            for _, parent in g.out_edges(nid):
                val_p = val * alpha
                # 取更大的值（因为可能从不同路径到达同一个父节点）
                if parent not in contrib or val_p > contrib[parent]:
                    contrib[parent] = val_p
                if parent not in visited:
                    visited.add(parent)
                    frontier.append((parent, val_p))
        return contrib
    
    # 计算每个疾病的语义贡献（可能对应多个MeSH术语，取最大值）
    disease_contrib = []
    for mesh_ids in name_to_mesh:
        agg = {}
        # 对每个疾病的每个MeSH ID，计算贡献并合并
        for mid in mesh_ids:
            c = semantic_contrib_from_concept(mid)
            for t, v in c.items():
                if t not in agg or v > agg[t]:
                    agg[t] = v
        disease_contrib.append(agg)



    # 计算IC（信息内容）

    rev_g = g.reverse()  # 反转图，变成 父->子，方便数子孙
    total_nodes_mesh = g.number_of_nodes()  # 分母变为全量节点数
    IC = {}
    for node in g.nodes():
        # 获取所有子孙节点
        descendants = nx.descendants(rev_g, node)
        # 计算基于拓扑的频率
        freq = (len(descendants) + 1.0) / total_nodes_mesh
        # 计算 IC
        IC[node] = -math.log(freq)


    # 计算相似度矩阵（DSS1和DSS2）
    n = len(disease_contrib)
    S1 = np.zeros((n, n), dtype=float)  # DSS1
    S2 = np.zeros((n, n), dtype=float)  # DSS2

    for i in range(n):

        di = disease_contrib[i]
        sum_i = sum(di.values()) if di else 0.0
        sum_i_w = sum(v * IC.get(t, 0.0) for t, v in di.items()) if di else 0.0
        
        for j in range(i, n):
            dj = disease_contrib[j]
            sum_j = sum(dj.values()) if dj else 0.0
            sum_j_w = sum(v * IC.get(t, 0.0) for t, v in dj.items()) if dj else 0.0
            
            # DSS1计算
            if sum_i + sum_j == 0.0:
                s1 = 0.0
            else:
                inter = set(di.keys()) & set(dj.keys())  # 共享术语
                shared = sum(di[t] + dj[t] for t in inter)
                s1 = shared / (sum_i + sum_j)
            
            # DSS2计算（IC加权）
            denom_w = (sum_i_w + sum_j_w)
            if denom_w == 0.0:
                s2 = 0.0
            else:
                inter = set(di.keys()) & set(dj.keys())
                shared_w = sum((di[t] + dj[t]) * IC.get(t, 0.0) for t in inter)

                # if a%100==1:
                #     print(f"{t}的ic值为:{IC.get(t, 0.0)}")


                s2 = shared_w / denom_w
            S1[i, j] = S1[j, i] = s1
            S2[i, j] = S2[j, i] = s2
    
    # 取平均
    S = (S1 + S2) / 2.0
    return S

import re
def main():

    # 1. 配置基本参数和文件路径
    MODE = 'mesh'
    BASE_DIR = '..'
    NAMES_CSV = 'new_disease_names.csv'
    OUT_CSV = 'd_ss.csv'

    # MeSH 数据和映射文件路径
    MESH_XML_PATH = './desc2025.xml'
    MESH_MAPPING_CSV = './match_mesh.csv' # 存储疾病名称到ID的映射关系
    MESH_EDGES_CSV = './mesh/tree_edges.csv' # 存储 MeSH DAG 边
    ALPHA = 0.5  # 语义贡献衰减因子
    
    # 2. 读取原始疾病名称列表
    names_csv_path = os.path.join(BASE_DIR, NAMES_CSV)
    df = pd.read_csv(names_csv_path)
    # 处理不同可能的列名或无列名情况
    if 'diseaseName' in df.columns:
        diseases = df['diseaseName'].astype(str).tolist()
    elif df.columns[0] == 'diseaseName':
        diseases = df.iloc[:, 0].astype(str).tolist()
    else:
        diseases = df.iloc[:, 0].astype(str).tolist()
    
    if MODE == 'mesh':
        xml_path = os.path.join(BASE_DIR, MESH_XML_PATH)
        mesh_mapping_csv = os.path.join(BASE_DIR, MESH_MAPPING_CSV)
        mesh_edges_csv = os.path.join(BASE_DIR, MESH_EDGES_CSV)
        
        # 检查必要的 XML 描述符文件是否存在
        if not os.path.isfile(xml_path):
            print(f'错误：找不到MeSH XML文件 {xml_path}')
            return
        
        # 3. 映射文件处理：如果 test_v2.csv 不存在，则执行第一次精准匹配
        if not os.path.isfile(mesh_mapping_csv):
            print('=' * 60)
            print('从MeSH XML提取描述符并构建映射索引...')

            # 解析 XML 构建 DAG 边和描述符列表
            desc_list, edges_set = parse_mesh_xml(xml_path,'D:\MiRnaProject\MDformer-main\HMDD4\semantic\supp2025.xml')
            
            # 建立名称/同义词到 MeSH UI 的反向索引
            name_to_ui = {}
            def norm(s: str) -> str:
                return ' '.join(s.lower().strip().split())
            
            for d in desc_list:
                for nm in [d['name']] + d['terms']:
                    key = norm(nm)
                    if key:
                        name_to_ui.setdefault(key, set()).add(d['ui'])
            
            # 为每个输入的疾病名称尝试精准匹配
            exact_matches = 0
            no_matches = 0
            ensure_dir(os.path.dirname(mesh_mapping_csv))
            with open(mesh_mapping_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['disease_name', 'mesh_ids'])
                writer.writeheader()
                for dname in diseases:
                    key = norm(dname)
                    mesh_ids = []
                    if key in name_to_ui:
                        mesh_ids = sorted(list(name_to_ui[key]))
                        exact_matches += 1
                    else:
                        no_matches += 1
                    writer.writerow({'disease_name': dname, 'mesh_ids': ';'.join(mesh_ids)})
            
            # 保存 MeSH 层级边到 CSV 文件
            ensure_dir(os.path.dirname(mesh_edges_csv))
            with open(mesh_edges_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['child', 'parent'])
                writer.writeheader()
                for c, p in sorted(edges_set):
                    writer.writerow({'child': c, 'parent': p})
            print('=' * 60)
        
        # 4. 读取现有的映射关系并检查缺失项
        mapping_rows = []
        with open(mesh_mapping_csv, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping_rows.append({'disease_name': row['disease_name'], 'mesh_ids': row.get('mesh_ids', '')})
        
        name_to_mesh = []
        for r in mapping_rows:
            ids = [t.strip() for t in r['mesh_ids'].split(';') if t.strip()]
            name_to_mesh.append(ids if ids else None)
        
        # 找出仍没有 MeSH ID 的疾病，并写入 missing_disease.txt
        missing = [diseases[i] for i, v in enumerate(name_to_mesh) if v is None or len(v) == 0]
        if missing:
            with open('missing_disease.txt', 'w', encoding='utf-8') as f:
                for i in missing:
                    f.write(i + '\n')

        # 如果存在缺失 ID，则中断程序，等待人工补全或补充匹配
        if len(missing) > 0:
            print(f'共有 {len(missing)} 个疾病未填写 MeSH ID，请查看: missing_disease.txt')
            print(f'请在补全 {mesh_mapping_csv} 后再次运行')
            return

        # 5. 读取边文件构建 DAG
        edges = []
        with open(mesh_edges_csv, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                c = row.get('child', '').strip()
                p = row.get('parent', '').strip()
                if c and p:
                    edges.append((c, p))
        
        # 6. 执行计算：计算基于贡献传播和 IC 加权的语义相似度
        S = compute_mesh_similarity(name_to_mesh, edges, alpha=ALPHA)
        
        # 7. 保存最终的相似度矩阵
        out_path = os.path.join(BASE_DIR, OUT_CSV)
        pd.DataFrame(S, index=diseases, columns=diseases).to_csv(out_path, index=False, float_format='%.8f',header=False)
        print(f'已成功生成疾病语义相似度矩阵: {out_path}')


if __name__ == '__main__':
    main()
