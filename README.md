<h1>🔅 Solar Equipment Maintenance Assistant (太陽能設備維運助手)</h1>
<h3>這是一套專為工業級太陽能設備運維 (O&M) 情境設計的「意圖驅動型檢索增強生成 (RAG) 多智能體協同系統」。透過創新的多智能體 (Multi-Agent) 解耦架構，本系統將零散的維修手冊與數據轉化為高精確度的維修 SOP 與動態流程圖，並具備最高權限的工安防線以杜絕 AI 幻覺。</h3>
Demo 影片: https://youtu.be/_J4XlbhsjIU 
<h2>📖 專案簡介 (Introduction)</h2>
<h3>動機</h3>
隨著全球再生能源基礎設施加速佈建，光電系統的穩定性直接關乎電網與投資回報率。然而，工業運維現場面臨手冊格式異質（PDF、CSV 等）、傳統搜尋難以精確撈取解法，且大型語言模型 (LLM) 在無約束下極易產生危及工安的「工業幻覺」。
<h3>目的</h3>
• 建構統一資料檢索：整合結構化與非結構化資料，並以高階光學與表格結構辨識技術無損還原設備參數。<br>
• Multi Agent 自動化生成：從意圖解構、混合重排到閉環遙測，提供從診斷到具體維護方案（含 Mermaid 流程圖）的全生命週期支援。<br>
• 絕對工安防線與優化維修速度：建立硬性防禦邊界避免盲目編造。實驗數據表明，本系統將 SOP 產出時間由 90 秒縮短至 40 秒，並將工安危害發生率降至 0%。<br>
<img width="1133" height="350" alt="image" src="https://github.com/user-attachments/assets/dbdd01ce-f3b8-44ca-a982-23bff1c2680b" />

## 🏗️ 系統架構 (Architecture)

本系統拋棄單一龐大提示詞的限制，採用**五大實體程式模組**之分散式協同架構：

1. **資料預處理與高階異質入庫管線 (`claudedata.py`)**：負責文件深度清洗、光學辨識、多層次語義切分與 LLM 自動化打標。
2. **意圖驅動混合檢索生成大腦 (`retrieval_brain.py`)**：最高推理中樞，結合元數據硬性錨定與軟性語義檢索，並設有防範幻覺的硬截斷防線。
3. **結構化流程圖轉譯代理 (`mermaid_agent.py`)**：利用動態 JSON 陣列萃取，將維修文本轉譯為無損渲染的 Mermaid.js 流程圖代碼。
4. **RAGAS 品質遙測裁判 (`score.py`)**：全無人值守之工業閉環反饋，針對知識落地性、回答相關性與工業安全性進行自動化稽核。
5. **全局路由與非阻塞異步調度主控台 (`app4.py`)**：整合生成器與多執行緒，釋放前端負載，達成人機互動的零延遲感。
<img width="345" height="561" alt="image" src="https://github.com/user-attachments/assets/0dc77263-1ab5-4443-a9c1-122ce76f13ce" />

---

## 📋 資料處理流程 (Data Processing Workflow)

本系統針對不同格式的資料來源設計了專屬的處理流水線，確保所有資訊都能被精確地檢索與利用 [cite: 50-62, 254-260]。

| 資料來源 (Data Source) | 處理階段 (Stage) | 執行動作與說明 (Action & Description) |
| :--- | :--- | :--- |
| **📄 PDF 文件**<br>(維修手冊、電氣接線圖) | **1. 高階光學與表格辨識** | 引入 `IBM Docling` 與 `TableFormerMode.ACCURATE` 模式，無損還原藏於複雜排版與表格死角的故障參數。 |
| | **2. 多層次標題語義切分** | 使用 `MarkdownHeaderTextSplitter` 錨定文件階層，再由 `RecursiveCharacterTextSplitter` 進行滾動切分，完美保留上下文從屬關係。 |
| | **3. LLM 自動化打標** | 調用 `Llama 3.2` 執行背景打標，強制輸出標準 JSON 作為詮釋資料 (Metadata) 寫入資料庫，建立強索引。 |
| **📊 CSV 表格**<br>(歷史維修日誌) | **1. 防禦性檔案解碼** | 實作 Fallback Encoding 算法，滾動嘗試多種解碼器解決亂碼。 |
| | **2. 缺失值前向填充** | 針對工業 CSV 常見的大區塊漏填，調用 `Pandas` 的 `ffill()` 動態補全組件名稱。 |
| **🔗 共同步驟** | **Embedding & Storage** | 文本強制加上 `search_document:` 前綴後，透過 `nomic-embed-text` 模型向量化，統一存入 **ChromaDB**。 |

---

## 👣 推理與檢索流程 (Inference & Retrieval)

當使用者輸入查詢（例如：「E-045」故障碼）時，系統執行以下多重檢索與防禦機制：

1. **意圖攔截 (Intent Interception)：** 獨立的意圖模型偵測到明確故障碼時，跳過模糊語義搜尋，直接調用 ChromaDB 的等值過濾器 (`$eq`)，消除相似型號干擾。
2. **混合重排 (Hybrid Search & Re-ranking)：** 在 Python 層級原生實作，結合語義檢索與正則表達式關鍵字詞頻統計，計算綜合評分。
3. **知識庫未命中截斷防線 (Critical Guardrail)：** 若最高分段落低於硬性閾值 (4.0)，系統立即中斷 (Bypass LLM)，直接輸出最高工安告示，100% 阻絕模型瞎編。
4. **生成與圖表轉譯：** LLM 整合上下文產出文字解決方案。同時，流程圖代理擷取動態 JSON，於後端渲染並針對「上鎖掛牌 (LOTO)」等高危險步驟掛上警示色彩。
<img width="477" height="335" alt="image" src="https://github.com/user-attachments/assets/5b75846c-f093-4784-8d45-30c0bf942f39" />

## 🛠️ 技術棧 (Tech Stack)

基於輕量化與隱私安全考量，本系統全面採用本機端部署與在地化模型架構：

| 類別 (Category) | 工具/庫 (Tools/Libraries) | 說明 (Description) |
| :--- | :--- | :--- |
| **LLM & Agent Core** | `Ollama`, `Llama 3.2`, `Gemma 3`, `Gemma 4` | Llama 3.2 負責打標與意圖推理；Gemma 3 負責 Mermaid 轉譯；Gemma 4 負責 RAGAS 稽核。 |
| **Data Extraction** | `IBM Docling`, `Pandas` | 高階光學字元與表格結構辨識管線，以及缺失值填充。 |
| **Embedding** | `nomic-embed-text` | 特徵對齊與文檔端向量化。 |
| **Vector Search** | `ChromaDB` | 提供元數據等值過濾的本地端向量資料庫。 |
| **UI & Async Flow** | `Gradio Blocks`, `Threading` | 全局路由、非阻塞異步調度架構，與 Base64 HTML Iframe 溯源技術。 |

---

## 最終預期結果:
文本輸出 : 
<img width="1560" height="641" alt="image" src="https://github.com/user-attachments/assets/6a2a6866-8c0d-4e98-bf2a-fa4acc9bf2d1" />
流程圖輸出 :
<img width="404" height="712" alt="image" src="https://github.com/user-attachments/assets/e11c1e6c-6b09-4432-965c-d7ebfe25ae95" />




