import xml.etree.ElementTree as ET
from fileinput import close

import pandas as pd
import os

from click import clear


def parse_mesh_supp_mapping(supp_xml_path):


    """
    解析MESH补充映射文件，生成从名称/同义词到唯一标识符的映射字典
    参数:
        supp_xml_path (str): MESH补充映射XML文件的路径
    返回:
        dict: 一个字典，键为名称/同义词(小写)，值为对应的唯一标识符集合
    """
    # 解析XML文件
    tree = ET.parse(supp_xml_path)
    root = tree.getroot()
    supp_mapping = {}

    # 遍历所有的SupplementalRecord节点
    # 遍历XML中的所有补充记录
    for record in root.findall('.//SupplementalRecord'):

        # 获取记录的唯一标识符(UI)
        ui_el = record.find('./SupplementalRecordUI')
        c_ui = ui_el.text.strip() if ui_el is not None else ""


        # 获取并处理主名称，转换为小写后添加到映射字典
        main_name=record.find('SupplementalRecordName/String').text.strip().lower()
        if main_name is not None:
            supp_mapping.setdefault(main_name, set()).add(c_ui)


        # 遍历概念列表中的每个概念
        for concept in record.findall('.//ConceptList/Concept'):
            # 获取并处理概念名称
            cn_el = concept.find('ConceptName/String')
            if cn_el is not None:
               supp_mapping.setdefault(cn_el.text.strip().lower(), set()).add(c_ui)


            # 获取并处理概念的所有同义词
            for term in concept.findall('.//TermList/Term/String'):
                if term.text:
                    supp_mapping.setdefault(term.text.strip().lower(), set()).add(c_ui)

    return supp_mapping


def process_unmatched_txt(txt_path, supp_xml_path, output_csv):

    #  解析字典
    supp_dict = parse_mesh_supp_mapping(supp_xml_path)



    # 疾病名称
    missing_diseases = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            missing_diseases.append(line.strip())

        f.close()

    xsx=[None]*len(missing_diseases)
    # 执行映射逻辑
    results = []
    found_count = 0
    i=0
    for dname in missing_diseases:
        norm_name = dname.lower()
        # 在 SCR 字典中查找
        mesh_id = supp_dict.get(norm_name, "")

        if mesh_id:
            mesh_id_str = ";".join(sorted(list(mesh_id)))

            xsx[i] = mesh_id_str
            found_count += 1
        i+=1

    for i,j in zip(missing_diseases,xsx):
        results.append({
            'disease_name': i,
            'mesh_ids': j
        })


    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False, encoding='utf-8')

    print("-" * 50)

    print(f"输入疾病总数: {len(missing_diseases)}")
    print(f"成功匹配: {found_count}")


process_unmatched_txt('missing_disease.txt', 'supp2025.xml', 'missing_dis_supp.csv')