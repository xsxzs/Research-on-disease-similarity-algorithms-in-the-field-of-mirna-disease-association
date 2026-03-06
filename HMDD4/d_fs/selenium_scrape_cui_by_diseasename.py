import pandas as pd
import time
import sys
import io
import locale
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

# Set encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=locale.getpreferredencoding(), errors='replace')

# ================= CONFIG =================
INPUT_FILE = 'disease_genes_second.csv'
OUTPUT_FILE = 'disease_genes_thired.csv'
CHROME_DRIVER_PATH = r'D:\gooledriver\chromedriver-win64\chromedriver.exe'
# ==========================================

def save_df(df, path):
    while True:
        try:
            df.to_csv(path, index=False, encoding='utf-8')
            return True
        except PermissionError:
            print(f"\n[ERROR] Cannot save to {path}. Please close Excel! Retrying in 5s...")
            time.sleep(5)

def main():
    print(f"Reading file: {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    
    # 确保有 cui 列
    if 'cui' not in df.columns:
        df['cui'] = "" # 默认为空字符串
        
    # 确保 cui 列是字符串类型，防止存入 NaN
    df['cui'] = df['cui'].astype(str).replace('nan', '')

    # 筛选目标：没有 genes 且 没有 cui 的行
    def is_target(row):
        has_gene = (not pd.isna(row.get('genes'))) and str(row.get('genes')).strip() != '' and str(row.get('gene_count')) != '0'
        has_cui = (not pd.isna(row.get('cui'))) and str(row.get('cui')).strip() != ''
        has_tag= (not pd.isna(row.get('tag_research'))) and str(row.get('tag_research')).strip() != ''
        # 如果已经有基因，或者已经有CUI，就跳过
        return (not has_gene) and (not has_cui) and (not has_tag)

    targets = [i for i, r in df.iterrows() if is_target(r)]
    print(f"Total diseases to scrape CUI: {len(targets)}")

    if not targets:
        print("All targets have CUIs or Genes. Nothing to do.")
        return

    # 启动浏览器
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1280,1024")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    service = Service(CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    print("\n" + "="*50)
    print("MANUAL SETUP PHASE (30 Seconds)")
    print("1. Log in to DisGeNET.")
    print("2. Navigate to: https://disgenet.com/advancedSearch")
    print("3. Ensure 'Disease Name' filter is selected.")
    print("Script will start in 30 seconds...")
    print("="*50 + "\n")

    try:
        driver.get("https://disgenet.com/advancedSearch")
    except: pass

    # 倒计时
    for i in range(30, 0, -1):
        print(f"Starting in {i} seconds...", end='\r')
        time.sleep(1)
    
    print("\n[START] CUI Scraping Started...")
    success_count = 0

    try:
        for i, idx in enumerate(targets):
            dname = str(df.at[idx, 'disease_name']).strip()
            print(f"[{i+1}/{len(targets)}] {dname:<30} ... ", end='', flush=True)

            try:
                wait = WebDriverWait(driver, 10)
                
                # 1. 尝试自动切换下拉框 (为了保险，每次都切一下)
                try:
                    filter_select = driver.find_element(By.NAME, "filter")
                    Select(filter_select).select_by_value("diseaseName")
                except:
                    pass # 如果找不到可能是页面还没加载好，或者是用户已经选好了

                # 2. 找输入框 (Textarea)
                search_box = None
                try:
                    search_box = driver.find_element(By.CSS_SELECTOR, "textarea.filter-textarea")
                except:
                    # 备用：placeholder
                    try:
                        search_box = driver.find_element(By.XPATH, "//textarea[contains(@placeholder, 'Enter')] ")
                    except:
                        pass
                
                if not search_box:
                    print(" [Error] Search box missing! Reloading...")
                    driver.get("https://disgenet.com/advancedSearch")
                    time.sleep(3)
                    continue #下一个病的循环 这个直接跳过


                # 3. 输入并搜索
                search_box.clear()
                driver.execute_script("arguments[0].value = '';", search_box)
                search_box.send_keys(dname)
                
                # 点击搜索
                try:
                    search_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Search')] ")
                    search_btn.click()
                except:
                    search_box.send_keys(Keys.ENTER)
                
                # 4. 等待结果并提取 CUI (表格第一行第二列)
                # 等待久一点，确保表格渲染
                time.sleep(6) 
                
                cui = None
                try:
                    # 定位 tbody 第一个 tr 的第二个 td
                    # 如果有多个结果，这里默认取第一个最匹配的
                    cui_element = driver.find_element(By.XPATH, "//tbody/tr[1]/td[2]")
                    raw_cui = cui_element.text.strip()
                    
                    if raw_cui and "Not available" not in raw_cui and re.match(r'C\d+', raw_cui):
                        cui = raw_cui
                except:
                    pass

                # 如果直接跳到了详情页 (URL变了)
                if not cui and "/browser/0/1/0/C" in driver.current_url:
                    match = re.search(r'C\d+', driver.current_url)
                    if match: cui = match.group(0)

                # 5. 保存结果
                if cui:
                    df.at[idx, 'cui'] = cui
                    success_count += 1
                    print(f" Found: {cui}")
                else:
                    # 标记为 "Not Found" 防止下次再查
                 
                    print(" Not Found")

                # 6. 重置回搜索页
                if "/browser/" in driver.current_url:
                    driver.back()
                    time.sleep(3)

                # 检查输入框是否还在，不在就刷新
                try:
                    driver.find_element(By.CSS_SELECTOR, "textarea.filter-textarea")
                except:
                    driver.get("https://disgenet.com/advancedSearch")
                    time.sleep(2)

            except Exception as e:
                print(f" Error: {e}")
                driver.get("https://disgenet.com/advancedSearch")
                time.sleep(2)

            # 每 5 条保存一次
            if (i+1) % 5 == 0:
                save_df(df, OUTPUT_FILE)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        driver.quit()
        save_df(df, OUTPUT_FILE)
        print(f"\nTask Finished. Total CUIs found: {success_count}")

if __name__ == "__main__":
    main()
