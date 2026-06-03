import pandas as pd
import json
import math

def convert_csv_to_jsonl(csv_path, jsonl_path):
    print(f"📂 正在讀取資料集：{csv_path}")
    
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='big5')
            
        df.columns = df.columns.str.strip()
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            count = 0
            for index, row in df.iterrows():
                device = str(row.get('設備', row.get('組件', '未知設備'))).strip()
                code = str(row.get('故障代碼', row.get('代碼', ''))).strip()
                issue = str(row.get('常見異常原因', row.get('異常問題', ''))).strip()
                loto = str(row.get('安全警告與隔離(LOTO)', '')).strip()
                check = str(row.get('檢查方式', '')).strip()
                solution = str(row.get('解決方法', '')).strip()

                if code == 'nan': code = ''
                if issue == 'nan': issue = '異常'

                instruction = f"請問{device}"
                if code: instruction += f"顯示代碼 {code}，"
                if issue: instruction += f"並且出現「{issue}」的狀況，"
                instruction += "該如何進行檢修？"
                
                output_parts = []
                if loto and loto != 'nan':
                    output_parts.append(f"🚨 【安全警告】：{loto}")
                if check and check != 'nan':
                    output_parts.append(f"🔍 【檢查步驟】：{check}")
                if solution and solution != 'nan':
                    output_parts.append(f"🛠️ 【解決方法】：{solution}")
                    
                output = "\n".join(output_parts)
                
                if len(output_parts) > 0:
                    json_record = {"instruction": instruction, "output": output}
                    f.write(json.dumps(json_record, ensure_ascii=False) + '\n')
                    count += 1
                    
        print(f"✅ 轉換成功！共產生 {count} 筆訓練資料，已儲存至 {jsonl_path}")
        
    except Exception as e:
        print(f"❌ 發生錯誤：{e}")

if __name__ == "__main__":
    convert_csv_to_jsonl('太陽能維修資料集.csv', 'solar_dataset.jsonl')
