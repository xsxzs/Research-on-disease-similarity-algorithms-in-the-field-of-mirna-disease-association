import csv
import os
import pandas as pd
import xml.etree.ElementTree as ET

# =================配置=================
INPUT_CSV = 'disease_genes_first.csv'
OUTPUT_CSV = 'disease_genes_second.csv'
DB_FILE = 'DisGeNET.txt'

DESC_XML = '../d_ss/desc2025.xml'
SUPP_XML = '../d_ss/supp2025.xml'

def load_enrichr_db(file_path):
 
    db = {}
    print(f"Loading Gene DB: {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            if len(parts) < 2: continue
            
            raw_name = parts[0]
            genes = parts[1:]
            
            db[raw_name.lower()] = {'genes': genes, 'db_name': raw_name}
    return db

def build_synonym_map(desc_path, supp_path):
   
    print("Parsing XMLs to build synonym map (This may take 1-2 mins)...")
    
    # key: any term (lower), value: list of all terms in that record (lower)
  
    syn_map = {}
    
    def parse_file(path, tag_record, tag_name, tag_term_list):
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return

        context = ET.iterparse(path, events=('end',))
        for event, elem in context:
            if elem.tag == tag_record:
                # 收集当前 Record 下的所有名字
                names = set()
                
                # Descriptor/Supplemental Name
                name_el = elem.find(f'./{tag_name}/String')
                if name_el is not None and name_el.text:
                    names.add(name_el.text.strip().lower())
                
                # Concept Lists (Entry Terms)
                for term in elem.findall(f'.//{tag_term_list}/String'):
                    if term.text:
                        names.add(term.text.strip().lower())
                
                # 将这个集合里的每个名字作为 Key，整个集合作为 Value 存进去
              
                names_tuple = tuple(names)
                for n in names:
                    syn_map[n] = names_tuple
                
                elem.clear() 

    # Desc
    # Structure: DescriptorRecord -> DescriptorName -> String
    #            ConceptList -> Concept -> TermList -> Term -> String
    parse_file(DESC_XML, 'DescriptorRecord', 'DescriptorName', 'Term')
    
    # Supp
    # Structure: SupplementalRecord -> SupplementalRecordName -> String
    #            ConceptList -> Concept -> TermList -> Term -> String
    parse_file(SUPP_XML, 'SupplementalRecord', 'SupplementalRecordName', 'Term')
    
    print(f"Synonym map built. Indexed {len(syn_map)} terms.")
    return syn_map

def main():
   
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    gene_db = load_enrichr_db(DB_FILE)
    synonyms = build_synonym_map(DESC_XML, SUPP_XML)
    
    df = pd.read_csv(INPUT_CSV)
    print(f"Processing {len(df)} rows...")
    
    recovered_count = 0
    
    # 新列：用来记录是用哪个同义词匹配上的
    match_notes = []

    # 2. 遍历处理
    for idx, row in df.iterrows():
        # 如果已经有基因了，跳过
        if pd.notna(row['genes']) and str(row['genes']).strip() != '':
            match_notes.append('original_exact')
            continue
        
        # 还没匹配上 -> 开始抢救
        dname = str(row['disease_name']).strip().lower()
        
        # 1. 查找同义词群
        # 也就是去 XML 里找这个名字属于哪个 Record，并拿到那个 Record 的所有名字
        if dname in synonyms:
            all_terms = synonyms[dname]
            
            found = False
            for term in all_terms:
                if term in gene_db:
                   
                    entry = gene_db[term]
                    
                    # 更新 DataFrame
              
                    df.at[idx, 'gene_count'] = len(entry['genes'])
                    df.at[idx, 'genes'] = ';'.join(entry['genes'])
                    # df.at[idx, 'matched_db_name'] = entry['db_name'] 
                    
                    match_notes.append(f"synonym: {term}")
                    recovered_count += 1
                    found = True
                    break 
            
            if not found:
                match_notes.append('failed_synonym_scan')
        else:
             match_notes.append('no_mesh_record_found')

    # 3. 保存
    df['match_note'] = match_notes
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    print("="*40)
    print(f"完成！")
    print(f"新增匹配: {recovered_count}")
    print(f"结果已保存至: {OUTPUT_CSV}")
    print("="*40)

if __name__ == "__main__":
    main()
