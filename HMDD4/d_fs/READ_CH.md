  第一阶段：获取疾病关联的基因 (Gene Mapping)
  这一阶段的目的是尽可能多地为你的疾病列表找到对应的基因。

   1. `match_genes_with_DisGeNET.py` (精准匹配)
      - 功能：这是第一波基础匹配。它读取输入文件，并读取本地的基因库文件
        DisGeNET.txt。通过疾病名称完全一致（忽略大小写）的原则，将基因匹配给疾病。
      - 输出：生成 disease_genes_first.csv，里面新增了 genes（基因列表）和 gene_count 列。

   2. `match_genes_with_synonyms.py` (同义词抢救匹配)
      - 功能：对于第一步没匹配上的疾病，这个脚本通过解析 MeSH 的 XML
        文件（desc2025.xml等）构建了一个庞大的“同义词字典”。如果疾病的原名找不到基因，它会去基因库里找这个疾病的各种同义词，只要撞上一个，就把基因赋给它。
      - 输出：生成 disease_genes_second.csv。
      

  第二阶段：通过网络爬虫/API 获取顽固疾病的基因
  对于本地数据库和同义词都搞不定的疾病，使用线上数据库 (DisGeNET),，但是无法直接使用疾病名查询gene，只能先使用疾病名称查询对应cui，再使用cui查询genes。

   5. `selenium_scrape_cui_by_diseasename.py` (通过疾病名称爬取 CUI)
      - 功能：Selenium 爬虫直接用 disease_name 在 DisGeNET网站搜索，尝试抓取对应的 cui。
      - 生成 disease_genes_thired.csv。

   6. `scrape_gene.py` (通过 API 获取基因)
      - 功能：前面拿到了 cui 后，这个脚本调用 DisGeNET 的官方 API。它遍历表格，凡是有 cui
        且还没拿到基因的疾病，就通过 API 查出其关联分数最高的 50 个基因填入表格。
      - 输出：生成 scrape_with_genes_final.csv 文件。此时，疾病已经全部拥有基因。

  第三阶段：获取基因的功能注释 (GO Terms)

  1.`Gene_Ontology.py` (获取 GO 注释)

- 功能：读取带有基因的表格（ scrape_with_genes_final.csv）。针对每一个疾病的基因列表，它调用 g:Profiler 的API，查询这些基因参与了哪些生物学过程（GO:BP）、分子功能（GO:MF）和细胞组分（GO:CC）。把所有基因的 GO ID
  汇总成一个集合，赋给该疾病。
- 输出：生成带有 go_terms 列的文件。disease_go.csv.

  第四阶段：计算相似性矩阵 (Similarity Calculation)

   2.`calculate_similarity.py` (核心计算)

- 功能：读取包含了疾病及其对应 go_terms 的文件（disease_go.csv）。
- 算法逻辑：
  - 把每个疾病的 GO terms 解析成集合。
  - 两两比较疾病。基础相似度使用 Jaccard 系数（交集除以并集）。
  - 引入了基于数量差异的指数衰减惩罚机制（math.exp(-alpha * 差异 / 最大值)）。意思是：如果疾病 A 有1000 个 GO 注释，疾病 B 只有 10 个，即使它们交集很高，由于基数相差太悬殊（可能 A 是研究很透彻的大类疾病，B是罕见病），脚本会强行压低它们的相似度得分，防止“假阳性”。
- 输出：生成最终的疾病功能相似性矩阵文件 d_fs.csv。

自动化流程：`run_pipeline.py`

