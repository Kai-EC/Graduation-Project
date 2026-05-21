# 05-FINAL 終極融合版：專業 UI 介面與強大 RAG 代理架構 (完美修復 LOG 表格高度版)
#目前有加使用指示05-14編輯確認版
import sys
import gradio as gr
import time
import base64
import os
import json
import re
import glob
import pandas as pd
from datetime import datetime
import threading

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.stdout.reconfigure(encoding='utf-8')

# --- 導入我們寫好的後端模組 ---
from score import Score_agent 
from retrieval_brain import MaintenanceRAG
from mermaid_agent import MermaidAgent
from claudedata import ingest_uploaded_files_to_chroma, delete_file_from_chroma
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- 基礎路徑設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR   = os.path.join(BASE_DIR, "chroma_db_unified")
EMBED_MODEL = "nomic-embed-text"
SAVE_DIR = os.path.join(BASE_DIR, "history_outputs")
EVAL_LOG_FILE = os.path.join(SAVE_DIR, "eval_logs.jsonl")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 實例化大腦代理 ---
rag     = MaintenanceRAG()
painter = MermaidAgent()
scorer  = Score_agent()

# ==========================================
# 1. CSS 樣式整合 (UI視覺 + Modal彈窗 + 黑字設定)
# ==========================================
custom_css = """
/* 基礎背景與字體 */
.gradio-container { background-color: #c9cfd6 !important; font-family: 'Noto Sans TC', 'Segoe UI', sans-serif; }

/* 主要繪圖與 SOP 顯示區塊 (網格奶茶色) */
.main-canvas {
    background-color: #faf0f0 !important;
    border: 1px solid #000000;
    border-radius: 12px;
    padding: 20px;
    min-height: 600px;
    background-image: radial-gradient(#cbd5e1 1px, transparent 0);
    background-size: 30px 30px;
    color: #000000 !important; 
}
.main-canvas * {
    color: #000000 !important; 
}

/* 左側檔案清單樣式 (修正跑版問題：改用 flex 佈局) */
.file-list-item { 
    margin-bottom: 8px; 
    padding: 12px; 
    background: #fff; 
    border-radius: 8px; 
    border-left: 5px solid #007AFF; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
    color: #000000;
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
}
.file-name-text {
    word-break: break-all; /* 允許長檔名自動換行 */
    margin-right: 15px;
    line-height: 1.4;
}
.status-badge {
    color: #22c55e;
    white-space: nowrap; /* 強制標籤不換行 */
    font-size: 0.9em;
    font-weight: bold;
}

.source-btn-row { margin-top: 10px; padding: 10px; background: #e2e8f0; border-radius: 8px; }

/* 表格黑字與淺灰白底設定 (修正深色模式覆蓋問題) */
.light-table, .light-table .table-wrap { 
    background-color: #e6ebf0 !important; 
}
.light-table table, .light-table th, .light-table td, .light-table tr, .light-table tbody, .light-table thead {
    background-color: #e6ebf0 !important; 
    color: #000000 !important; 
    border-color: #c9cfd6 !important;
}
.light-table span {
    color: #000000 !important; 
}

/* 🌟 強制穿透拉長表格內部容器 (打破 Gradio 預設的限制) */
.tall-table {
    height: 800px !important;
}
.tall-table .wrap, 
.tall-table .table-wrap, 
.tall-table .tbody-wrap, 
.tall-table table {
    max-height: 800px !important;
    height: 100% !important;
}

/* 按鈕自訂顏色 */
button.primary { background-color: #dbb98a !important; background: #dbb98a !important; border: 1px solid #c4a47a !important; color: white !important; }
button.primary:hover { background-color: #c4a47a !important; }
#upload_btn { background-color: #8db0b0 !important; background: #8db0b0 !important; border: 1px solid #7a9999 !important; color: white !important; }
#upload_btn:hover { background-color: #7a9999 !important; }
.vertical-btn-group button { margin-bottom: 5px !important; width: 100% !important; }

/* 歷史詳情 Modal */
.detail-modal-content { background: white; padding: 10px; border: 2px solid #007AFF; border-radius: 15px; margin-top: 20px; color: #000000; }

/* --- 來源彈窗與導覽彈窗專屬 CSS --- */
.modal-toggle{display:none;}
.modal-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(15,23,42,.75);z-index:99999;display:flex;justify-content:center;align-items:center;opacity:0;visibility:hidden;backdrop-filter:blur(4px);}
.modal-toggle:checked ~ .modal-overlay{opacity:1;visibility:visible;}
.modal-container{background:#fff;width:92%;max-width:1100px;height:88vh;border-radius:14px;overflow:hidden;display:flex;flex-direction:column;}
.modal-header{background:#1e293b;padding:16px;display:flex;align-items:center;gap:12px;}
.modal-title{color:#f1f5f9;flex:1;font-weight:bold;}
.modal-badge{color:#fff;padding:3px 10px;border-radius:20px;font-size:0.8em;}
.modal-close{color:#94a3b8;font-size:1.5rem;cursor:pointer;}
.modal-content{flex:1;overflow-y:auto;background:#f8fafc; color:#000000; padding:15px;}
.source-btn{display:inline-flex;padding:10px 18px;background:#1e293b;border-radius:9px;color:#fff;cursor:pointer;margin:5px 5px 0 0;}
.source-btn:hover{background:#3b82f6;}

/* --- 系統導覽問號按鈕 --- */
.help-icon-btn {
    background-color: #dbb98a;
    color: white;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    cursor: pointer;
    font-size: 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.15);
    transition: all 0.2s ease;
}
.help-icon-btn:hover {
    background-color: #c4a47a;
    transform: scale(1.1);
}
"""

# ==========================================
# 2. 輔助函式與遙測紀錄
# ==========================================
def sanitize_filename(text): return re.sub(r'[^\w\u4e00-\u9fa5]', '_', text)[:20]

def save_to_local(query, answer, html_content, source_html):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query  = sanitize_filename(query)
    base_filename = f"{timestamp}_{safe_query}"

    json_path = os.path.join(SAVE_DIR, f"{base_filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"timestamp": timestamp, "query": query, "answer_text": answer}, f, ensure_ascii=False, indent=4)
        
    full_combined_html = f"{html_content}\n<div style='margin-top:30px; border-top:2px dashed #ccc; padding-top:20px;'>{source_html}</div>"
    html_path = os.path.join(SAVE_DIR, f"{base_filename}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_combined_html)

def log_evaluation(query, eval_result):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "groundedness": eval_result.get("groundedness", "Fail"),
        "relevance": eval_result.get("relevance", "Fail"),
        "safety": eval_result.get("safety", "Fail"),
        "reasoning": eval_result.get("reasoning", ""),
        "suggestion": eval_result.get("suggestion", "")
    }
    with open(EVAL_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def get_eval_logs():
    if not os.path.exists(EVAL_LOG_FILE):
        return pd.DataFrame(columns=["紀錄時間", "使用者查詢", "知識落地性", "回答相關性", "工業安全性", "判定原因", "系統優化建議"])
    logs = []
    with open(EVAL_LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                logs.append([
                    data.get("timestamp", ""), data.get("query", ""),
                    data.get("groundedness", ""), data.get("relevance", ""), data.get("safety", ""),
                    data.get("reasoning", ""), data.get("suggestion", "")
                ])
            except: pass
    return pd.DataFrame(logs[::-1], columns=["紀錄時間", "使用者查詢", "知識落地性", "回答相關性", "工業安全性", "判定原因", "系統優化建議"])

def get_history_dataframe():
    files = glob.glob(os.path.join(SAVE_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    data = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                j = json.load(file)
                ts = j.get("timestamp", "")
                if len(ts) == 15: # YYYYMMDD_HHMMSS
                    date_str = f"{ts[:4]}/{ts[4:6]}/{ts[6:8]}"
                    time_str = f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
                else:
                    date_str, time_str = "-", "-"
                
                query = j.get("query", "無摘要")
                file_id = os.path.basename(f).replace('.json', '')
                data.append([date_str, time_str, query, file_id])
        except: pass
        
    if not data:
        return pd.DataFrame(columns=["📅 日期", "⏰ 時間", "💬 問題內容摘要", "隱藏ID"])
    return pd.DataFrame(data, columns=["📅 日期", "⏰ 時間", "💬 問題內容摘要", "隱藏ID"])

def load_history_detail(evt: gr.SelectData, df):
    row_idx = evt.index[0]
    record_id = df.iloc[row_idx, 3] # 隱藏ID在第4欄
    html_path = os.path.join(SAVE_DIR, f"{record_id}.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f: raw_html = f.read()
        b64 = base64.b64encode(raw_html.encode('utf-8')).decode('utf-8')
        iframe = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="900px" style="border:none; border-radius:15px;"></iframe>'
        return iframe, gr.update(visible=True), gr.update(visible=True)
    return "<div style='color:red; padding:20px;'>找不到對應的檔案紀錄。</div>", gr.update(visible=True), gr.update(visible=True)

# 將左側 DB 狀態轉換為 UI 樣式的 HTML (修正跑版)
# 將左側 DB 狀態轉換為 UI 樣式的 HTML
def get_db_files_html():
    base_html = """<div><h3 style="margin-top: 0px; color:#000;">📁 資料庫狀態</h3>"""
    # 檢查 rag 實例中是否已經有成功建立的 vectorstore
    if getattr(rag, 'vectorstore', None) is None: 
        return base_html + "<div style='color:#ef4444;'>尚未建立或載入資料庫</div></div>"
        
    try:
        # 直接使用 rag.vectorstore，不要再次呼叫 Chroma(...)
        metadatas = rag.vectorstore.get(include=["metadatas"]).get("metadatas", [])
        sources = sorted({m["source"] for m in metadatas if m and "source" in m})
        
        if not sources: 
            return base_html + "<div style='color:#64748b;'>資料庫為空</div></div>"
        
        for s in sources:
            base_html += f"""<div class="file-list-item"><div class="file-name-text">📄 {s}</div><div class="status-badge">● 運作中</div></div>"""
        return base_html + "</div>"
    except Exception as e: 
        return base_html + f"<div style='color:#ef4444;'>讀取失敗: {str(e)}</div></div>"

def get_file_choices():
    # 同樣直接使用 rag.vectorstore
    if getattr(rag, 'vectorstore', None) is None: 
        return []
    try:
        metadatas = rag.vectorstore.get(include=["metadatas"]).get("metadatas", [])
        return sorted({m["source"] for m in metadatas if m and "source" in m})
    except: 
        return []

# ==========================================
# 3. Mermaid 與彈跳來源渲染 HTML
# ==========================================
def generate_mermaid_html(answer_text, mermaid_code):
    clean_code = mermaid_code.replace('```mermaid', '').replace('```', '').strip()
    html_content = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="UTF-8"><script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script><style>body {{ font-family: sans-serif; padding:10px; margin:0; color:#000000; }}.mermaid {{ display:flex; justify-content:center; }}</style></head><body><div class="mermaid">{clean_code}</div><script>mermaid.initialize({{ startOnLoad:true, theme:'neutral', securityLevel:'loose' }}); mermaid.contentLoaded();</script></body></html>"""
    b64 = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
    iframe_html = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="800px" style="border:none;"></iframe>'
    return iframe_html, html_content

def _get_ai_tags_html(meta: dict) -> str:
    return f"""<div style="margin-top: 15px; padding: 10px; background: #f0fdf4; border-radius: 8px; border: 1px solid #bbf7d0;"><div style="font-size: 0.85em; color: #166534; font-weight: bold; margin-bottom: 5px;">🤖 AI 語意解析標籤</div><div style="display: flex; gap: 8px; flex-wrap: wrap;"><span style="background:#dbeafe; color:#1e40af; padding:4px 10px; border-radius:12px; font-size:0.85em;">🏷️ 設備: {meta.get("ai_equipment", "未知")}</span><span style="background:#fee2e2; color:#991b1b; padding:4px 10px; border-radius:12px; font-size:0.85em;">⚠️ 故障碼: {meta.get("ai_error_code", "無")}</span><span style="background:#f3e8ff; color:#6b21a8; padding:4px 10px; border-radius:12px; font-size:0.85em;">📁 分類: {meta.get("ai_category", "未知")}</span></div></div>"""

def _build_pdf_viewer(doc, source: str) -> tuple:
    page_no = doc.metadata.get("page_number", "未知")
    viewer_html = f"""<div style="padding:20px; line-height:1.6; color:#000000;"><div style="background:#fff3f3; border-left:5px solid #ef4444; padding:10px;"><h4 style="color:#000000; margin:0;">📄 來源檔案：{source} (第 {page_no} 頁)</h4></div>{_get_ai_tags_html(doc.metadata)}<h3 style="margin-top:20px; border-bottom:2px solid #10b981; color:#000000;">🔍 擷取文本內容</h3><div style="background:white; border:1px solid #e2e8f0; padding:15px; border-radius:8px; white-space:pre-wrap; color:#000000;">{doc.page_content}</div></div>"""
    return f"第 {page_no} 頁", viewer_html

def _build_csv_viewer(doc) -> tuple:
    meta = doc.metadata
    display_row = meta.get("row_index", 0) + 2
    columns = json.loads(meta.get("columns_json", "[]")) if "columns_json" in meta else []
    row_dict = json.loads(meta.get("row_json", "{}")) if "row_json" in meta else {}
    table_rows_html = "".join([f'<tr><td style="background:#f1f5f9; padding:10px; border:1px solid #e2e8f0; color:#000000;">{col}</td><td style="padding:10px; border:1px solid #e2e8f0; color:#000000;">{row_dict.get(col, "—")}</td></tr>' for col in columns])
    viewer_html = f"""<div style="padding:24px; background:#f8fafc; min-height:100%; color:#000000;">{_get_ai_tags_html(meta)}<h3 style="margin-top:20px; border-bottom:2px solid #3b82f6; color:#000000;">📋 原始資料（第 {display_row} 行）</h3><table style="width:100%; border-collapse:collapse; color:#000000;">{table_rows_html}</table><h3 style="margin-top:20px; border-bottom:2px solid #10b981; color:#000000;">🔍 語義化內容</h3><div style="background:white; border:1px solid #e2e8f0; padding:15px; border-radius:8px; color:#000000;">{doc.page_content}</div></div>"""
    return f"第 {display_row} 行", viewer_html

def build_source_html(message: str) -> str:
    if not rag.vectorstore: return "<p style='color:#ef4444;'>無連接資料庫，無法提供來源。</p>"
    docs = rag.vectorstore.similarity_search(message, k=2) 
    html = """<div class="source-btn-row"><p style="margin:0 0 5px 0; font-size:0.9em; font-weight:bold; color:#0f172a;">🔍 系統檢索來源 (點擊展開)</p><div style="display:flex; flex-wrap:wrap;">"""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知檔案")
        file_type = source.lower().rsplit(".", 1)[-1]
        uid = f"modal-{int(time.time()*1000)}-{i}"
        loc, viewer = _build_pdf_viewer(doc, source) if file_type != "csv" else _build_csv_viewer(doc)
        bg = "#ef4444" if file_type=="pdf" else "#10b981" if file_type=="csv" else "#2563eb"
        html += f"""<div><label for="{uid}" class="source-btn">📄 [{i+1}] {source}</label><input type="checkbox" id="{uid}" class="modal-toggle"><div class="modal-overlay"><div class="modal-container"><div class="modal-header"><div class="modal-title">📄 {source}</div><span class="modal-badge" style="background:{bg};">{file_type.upper()}</span><label for="{uid}" class="modal-close">&times;</label></div><div class="modal-content">{viewer}</div></div></div></div>"""
    return html + "</div></div>"

# ==========================================
# 4. 系統核心邏輯對接
# ==========================================
def handle_query(message, history):
    # 1. 立即更新 UI，顯示「正在思考」並清空輸入框
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⏳ 正在檢索技術手冊並生成文字 SOP..."})
    yield "", history, "<div class='main-canvas' style='text-align:center; padding-top:200px; color:#64748b;'>等待圖表生成...</div>", "生成中...", "<p style='color:#64748b;'>檢索中...</p>"

    # 2. 執行 RAG 檢索 (耗時)
    raw_answer = rag.search_and_reason(message)
    answer     = raw_answer.replace("ตรวจสอบ", "檢查").replace("ตรวจ", "檢")
    source_html = build_source_html(message)

    # 更新 UI：文字 SOP 已完成，提示正在畫圖
    history[-1]["content"] = "⏳ 文字 SOP 產出完畢。正在分析邏輯並繪製流程圖..."
    yield "", history, "<div class='main-canvas' style='text-align:center; padding-top:200px; color:#64748b;'>等待圖表生成...</div>", answer, source_html

    # 3. 執行 Mermaid 繪圖 (耗時)
    vis_html, chart_code = "", ""
    if any(k in answer for k in ["檢查", "步驟", "解決", "流程", "分析"]):
        chart_code = painter.generate(answer)
        vis_html, raw_html_to_save = generate_mermaid_html(answer, chart_code)
    else:
        vis_html = "<div style='text-align:center; padding-top:200px; color:#64748b; font-size:1.2em;'>無流程邏輯，請直接參考右方文字 SOP。</div>"
        raw_html_to_save = "<h2>無流程圖</h2>"

    # 最終更新 UI：全部完成
    history[-1]["content"] = "✅ 分析已完成！請檢視中間的『文字 SOP』與『流程圖』，並可點擊下方來源查閱原始技術手冊。"
    yield "", history, vis_html, answer, source_html

    # 4. 將存檔與 RAGAS 評分丟到背景執行，絕對不阻塞前端 UI
    def background_tasks(msg, ans, html_save, src_html, code):
        try:
            save_to_local(msg, ans, html_save, src_html)
            eval_result = scorer.generate(query=msg, text=ans, mermaid_code=code)
            log_evaluation(msg, eval_result)
        except Exception as e:
            print(f"背景任務發生錯誤: {e}")

    threading.Thread(
        target=background_tasks, 
        args=(message, answer, raw_html_to_save, source_html, chart_code)
    ).start()

def handle_file_deletion(filename):
    if not filename: return get_db_files_html(), gr.update(choices=get_file_choices()), gr.update(value=None)
    delete_file_from_chroma(filename)
    rag.reload()
    return get_db_files_html(), gr.update(choices=get_file_choices()), gr.update(value=None)

def process_uploaded_files(file_objs):
    if not file_objs: return get_db_files_html(), gr.update(choices=get_file_choices())
    ingest_uploaded_files_to_chroma([f.name for f in file_objs])
    rag.reload() 
    return get_db_files_html(), gr.update(choices=get_file_choices())

def export_maintenance_report(sop_text):
    if not sop_text: return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"Solar_SOP_Report_{timestamp}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== ☀️ 太陽能設備維修 SOP 報告 ===\n\n")
        f.write(sop_text)
    return file_path

# ==========================================
# 5. UI 介面建置 
# ==========================================
# 定義包含標題與隱藏式導覽視窗的 HTML
help_header_html = """
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
    <h1 style="margin: 0; font-size: 2.2em; color: #1e293b;">☀️ 太陽能設備維修智慧助手</h1>
    <label for="help-modal-toggle" class="help-icon-btn" title="系統使用說明">?</label>
</div>

<input type="checkbox" id="help-modal-toggle" class="modal-toggle">
<div class="modal-overlay">
    <div class="modal-container" style="max-width: 650px; height: auto; max-height: 85vh;">
        <div class="modal-header" style="background: #dbb98a;">
            <div class="modal-title" style="color: #fff; font-size: 1.2em;">📖 系統使用流程指示</div>
            <label for="help-modal-toggle" class="modal-close" style="color: #fff;">&times;</label>
        </div>
        <div class="modal-content" style="color: #333; line-height: 1.8; padding: 25px;">
            <h3 style="border-bottom: 2px solid #dbb98a; padding-bottom: 8px; margin-top: 0; color:#1e293b;">1. 📂 準備知識庫</h3>
            <p style="margin-top: 5px; margin-bottom: 20px;">在左側面板上傳太陽能設備的技術手冊、維修紀錄等文件。點擊「上傳並更新大腦」後，系統會自動處理並將知識存入 AI 資料庫（顯示綠色燈號）。</p>
            
            <h3 style="border-bottom: 2px solid #dbb98a; padding-bottom: 8px; color:#1e293b;">2. 💬 提出維修問題</h3>
            <p style="margin-top: 5px; margin-bottom: 20px;">在右側對話框輸入您遇到的故障情況（例如：「逆變器亮紅燈，顯示 E-045」），點擊發送。</p>
            
            <h3 style="border-bottom: 2px solid #dbb98a; padding-bottom: 8px; color:#1e293b;">3. 📊 檢視分析結果</h3>
            <p style="margin-top: 5px; margin-bottom: 20px;">系統將在中央面板為您生成<b>文字 SOP</b> 與 <b>動態流程圖</b>。您也可以點擊對話框下方的「檢索來源」按鈕，展開查閱原始技術手冊的段落出處。</p>
            
            <h3 style="border-bottom: 2px solid #dbb98a; padding-bottom: 8px; color:#1e293b;">4. 🕵️ 歷史與工程監控</h3>
            <p style="margin-top: 5px; margin-bottom: 0;">您可以隨時切換至上方分頁，檢視過去的<b>歷史紀錄</b>，或進入 <b>LOG 評分日誌</b> 監控 AI 回答的知識落地性與工業安全合規程度。</p>
        </div>
    </div>
</div>
"""

with gr.Blocks(title="Solar AI Assistant Pro", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.HTML(help_header_html)
    with gr.Tabs():
        
        # ── 第一分頁：主系統 ──
        with gr.TabItem("🔧 AI 維修助手"):
            with gr.Row(equal_height=True):
                
                # 左側：檔案管理
                with gr.Column(scale=3, variant="panel"):
                    db_files_display = gr.HTML(value=get_db_files_html())
                    file_input = gr.File(label="上傳新文件", file_count="multiple", file_types=[".csv", ".pdf", ".docx", ".txt"])
                    delete_dropdown = gr.Dropdown(choices=get_file_choices(), label="選擇要刪除的檔案", show_label=False)
                    
                    with gr.Column(elem_classes=["vertical-btn-group"]):
                        up_btn = gr.Button("🚀 上傳並更新大腦", variant="primary", size="sm", elem_id="upload_btn")
                        del_btn = gr.Button("🗑️ 刪除所選檔案", variant="stop", size="sm")

                # 中間：輸出顯示框
                with gr.Column(scale=12):
                    with gr.Tabs():
                        with gr.TabItem("📄 文字 SOP"):
                            with gr.Column(elem_classes=["main-canvas"]):
                                sop_display = gr.Markdown("請在右側輸入問題以生成詳細 SOP...")
                        with gr.TabItem("🖼️ 建議維修流程圖"):
                            canvas_display = gr.HTML(
                                value="<div style='text-align:center; padding-top:200px; color:#64748b;'>等待 AI 分析數據並生成流程圖...</div>",
                                elem_classes=["main-canvas"]
                            )
                    with gr.Row():
                        export_btn = gr.Button("📥 下載文字報告 (.txt)", variant="secondary")
                        file_output = gr.File(label="下載您的報告", visible=False)

                # 右側：AI 對話框
                with gr.Column(scale=4):
                    gr.Markdown("### 💬 AI 分析助理")
                    chatbot = gr.Chatbot(height=450, label="互動視窗")
                    msg = gr.Textbox(placeholder="描述故障情況 (例:逆變器亮紅燈)...", show_label=False)
                    with gr.Row():
                        submit_btn = gr.Button("發送問詢", variant="primary")
                        clear_btn = gr.Button("清除對話")
                    
                    source_display = gr.HTML('<div class="source-btn-row"><p style="margin:0; font-size:0.9em; color:#64748b;">等待檢索來源...</p></div>')

        # ── 第二分頁：歷史紀錄 ──
        with gr.TabItem("🕒 歷史紀錄"):
            gr.Markdown("### 🗂️ 全域歷史問詢追蹤")
            with gr.Row(): refresh_history = gr.Button("🔄 刷新歷史紀錄表單", variant="primary")
            
            history_df = gr.Dataframe(value=get_history_dataframe(), interactive=False, elem_classes=["light-table", "tall-table"])
            
            history_detail_area = gr.HTML(visible=False, elem_classes=["detail-modal-content"])
            close_detail_btn = gr.Button("關閉詳細紀錄", visible=False, variant="stop")
            
        # ── 第三分頁：LOG 評分 ──
        with gr.TabItem("📊 LOG 評分日誌"):
            gr.Markdown("### 🕵️ 工程師監控面板 (RAGAS 遙測日誌)")
            with gr.Row(): refresh_log = gr.Button("🔄 抓取最新遙測日誌", variant="primary")
            
            log_table = gr.Dataframe(
                value=get_eval_logs(), 
                interactive=False, 
                wrap=True,
                elem_classes=["light-table", "tall-table"]
            )

    # ==========================================
    # 6. 事件綁定 
    # ==========================================
    msg.submit(handle_query, [msg, chatbot], [msg, chatbot, canvas_display, sop_display, source_display])
    submit_btn.click(handle_query, [msg, chatbot], [msg, chatbot, canvas_display, sop_display, source_display])
    clear_btn.click(lambda: (None, "", "<div class='main-canvas'></div>", "已清除"), None, [chatbot, msg, canvas_display, sop_display])
    
    up_btn.click(process_uploaded_files, file_input, [db_files_display, delete_dropdown])
    del_btn.click(handle_file_deletion, delete_dropdown, [db_files_display, delete_dropdown, delete_dropdown])
    
    export_btn.click(lambda s: gr.update(value=export_maintenance_report(s), visible=True), sop_display, file_output)
    
    refresh_history.click(get_history_dataframe, None, history_df)
    refresh_log.click(get_eval_logs, None, log_table)
    
    history_df.select(load_history_detail, history_df, [history_detail_area, history_detail_area, close_detail_btn])
    close_detail_btn.click(lambda: (gr.update(visible=False), gr.update(visible=False)), None, [history_detail_area, close_detail_btn])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", theme=gr.themes.Soft(primary_hue="blue"), allowed_paths=[DATA_DIR], css=custom_css, share=False)