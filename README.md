<h1>🔅 Solar Equipment Maintenance Assistant (太陽能設備維運助手)</h1>
<h3>這是一個具備跨資料檢索與推理能力的太陽能設備維運系統，透過階層式 RAG 架構與生成式 AI，針對其中設備故障或損壞時，將零散的維修手冊與數據轉化為標準化的維修方案(發生原因、解決方式)與動態流程圖。</h3>
<h2>📖 專案簡介 (Introduction)</h2>
<h3>動機</h3>
隨著綠色能源轉型，大規模太陽能發電廠的穩定性至關重要。然而，現有的維運（O&M）流程面臨資料零散、依賴人工經驗以及跨系統檢索低效的問題 。
<h3>目的</h3>
• 建構統一資料檢索：整合結構化（CSV）與非結構化（PDF）資料<br>
• 實現自動化生成：從問題診斷到生成具體的維護方案<br>
• 優化維修速度：縮短故障診斷週期，減少發電損失<br>
<img width="1133" height="350" alt="image" src="https://github.com/user-attachments/assets/dbdd01ce-f3b8-44ca-a982-23bff1c2680b" />

## 🏗️ 系統架構 (Architecture)

本系統分為「資料前處理」與「推理解索」兩大階段：

1.  **資料層**：透過 OCR 與語意化技術處理原始文件，並存入向量資料庫。
2.  **檢索層**：使用者提問後，系統透過 Embedding 模型進行語意比對，檢索相關文檔。
3.  **生成層**：LLM (Llama/Gemma) 整合檢索到的上下文，生成文字回答與圖表代碼。
<img width="1116" height="552" alt="image" src="https://github.com/user-attachments/assets/55fb3ab5-00ed-43c8-af46-5e624f794c0f" />

---

## 📋 資料處理流程 (Data Processing Workflow)

本系統針對不同格式的資料來源設計了專屬的處理流水線，確保所有資訊都能被精確地檢索與利用 [cite: 50-62, 254-260]。

| 資料來源 (Data Source) | 處理階段 (Stage) | 執行動作與說明 (Action & Description) |
| :--- | :--- | :--- |
| **📄 PDF 文件**<br>(維修手冊、技術文件) | **1. OCR 光學辨識** | 透過 `EasyOCR` 將掃描檔或圖片中的文字、表格轉換為數位文字。 |
| | **2. Markdown Chunking** | 將 OCR 輸出轉換為 Markdown，並將長文章切分成具備上下文資訊的連貫小段落 (Chunks)。 |
| | **3. Metadata 封裝** | 將切割後的文本封裝為 Document 物件，並標註來源資訊（如：檔名、頁碼）。 |
| **📊 CSV 表格**<br>(歷史維修數據) | **1. 語意化格式 (Semantization)** | 將結構化數據轉換為自然語言描述。例如：將 `{"Component": "A"}` 轉換為「組件 A 發生錯誤...」。 |
| | **2. JSON 轉換** | 將每一行 (Row) 視為獨立事件，轉換為 JSON 格式，方便程式解耦與後續檢視。 |
| | **3. Metadata 封裝** | 同樣將 JSON 資料封裝為 Document 物件，並加入對應的 Metadata 標籤。 |
| **🔗 共同步驟** | **Embedding & Storage** | 所有 Document 物件經由 Embedding 模型向量化後，統一建立 **FAISS Index**。 |

---

## 👣 推理與檢索流程 (Inference & Retrieval)

當使用者輸入查詢（例如：「逆變器母線過熱」）時，系統執行以下步驟 ：<br>
第一層檢索：識別設備類別（如：逆變器、光伏板）。<br>
第二層檢索：診斷故障原因（如：過壓保護失效、溫度異常）。<br>
第三層檢索：提取維修方式（如：檢測電壓、元件更換）。<br>
LLM 生成：整合上下文，生成文字解決方案 + Mermaid 流程圖代碼。<br>
<img width="477" height="335" alt="image" src="https://github.com/user-attachments/assets/5b75846c-f093-4784-8d45-30c0bf942f39" />

## 🛠️ 技術棧 (Tech Stack)

| 類別 (Category) | 工具/庫 (Tools/Libraries) | 說明 (Description) |
| :--- | :--- | :--- |
| **Language** | Python 3.x | 核心開發語言 |
| **OCR & PDF** | `easyocr`, `PyMuPDF` (fitz), `pypdf` | 處理 PDF 文件與光學字元辨識 |
| **Embedding** | `sentence-transformers` | 使用 `all-MiniLM-L6-v2` 模型生成向量 |
| **Vector Search** | `faiss-cpu` | 高效能向量檢索 (Vector Indexing) |
| **Data Logic** | `numpy`, `re` (Regex) | 數據運算與文本切塊 (Text Chunking) |
| **GenAI Model** | Llama 3.2 / Gemma 3 | (整合端)負責生成最終自然語言回應與圖表代碼|

---

## 最終預期結果:
文本輸出 : 
<img width="1560" height="641" alt="image" src="https://github.com/user-attachments/assets/6a2a6866-8c0d-4e98-bf2a-fa4acc9bf2d1" />
流程圖輸出 :
<img width="404" height="712" alt="image" src="https://github.com/user-attachments/assets/e11c1e6c-6b09-4432-965c-d7ebfe25ae95" />




