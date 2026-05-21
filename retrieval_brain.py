# 04-XX 零依賴升級版：動態意圖攔截 (Intent Parsing) + Metadata 精準過濾 + 原生重排序
# 核心優化：修正 Chroma 欄位過濾器、Nomic 查詢前綴注入、防止未命中硬猜的防線限制
import os
import re
import json
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

class MaintenanceRAG:
    def __init__(self, model_name="llama3.2", embed_model="nomic-embed-text", top_k=4):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.BASE_DIR, "chroma_db_unified")
        
        self.top_k = top_k
        self.embeddings = OllamaEmbeddings(model=embed_model)
        
        self.llm = OllamaLLM(model=model_name, temperature=0)
        self.intent_llm = OllamaLLM(model="llama3.2", temperature=0)
        
        self.vectorstore = None
        self.reload()

    def reload(self):
        """重新載入資料庫"""
        if os.path.exists(self.db_path):
            self.vectorstore = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            print("[系統日誌] 🧠 RAG Brain: 向量資料庫載入成功！(零依賴安全模式)")
        else:
            self.vectorstore = None
            print(f"⚠️ 警告: 找不到資料庫 {self.db_path}，請先上傳檔案或執行預處理。")

    def _extract_intent(self, query: str) -> dict:
        """🌟 核心機制 1：意圖分析器（修正對齊打標鍵名）"""
        prompt = ChatPromptTemplate.from_template("""
        你是一個維修意圖分析器。請從使用者的問題中提取「設備名稱」與「故障代碼/警報碼」。
        如果沒有明確提及，請填寫 "無"。
        【嚴格限制】：只能輸出純 JSON，不可有任何 Markdown 或其他文字。
        
        JSON 格式：{{"ai_equipment": "...", "ai_error_code": "..."}}
        
        使用者問題：{query}
        """)
        
        try:
            chain = prompt | self.intent_llm
            response = chain.invoke({"query": query})
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                raw_json = json.loads(match.group(0))
                return {
                    "ai_equipment": raw_json.get("ai_equipment", "無"),
                    "ai_error_code": raw_json.get("ai_error_code", "無")
                }
        except Exception as e:
            print(f"[系統警告] 意圖解析失敗，將退回全域檢索: {e}")
            
        return {"ai_equipment": "無", "ai_error_code": "無"}

    def search_and_reason(self, query):
        if not self.vectorstore:
            return "錯誤：資料庫未初始化，請先上傳維修手冊。"

        # ==========================================================
        # 🌟 階段 1：意圖攔截與動態過濾 (注入 Nomic 查詢前綴)
        # ==========================================================
        intent = self._extract_intent(query)
        target_code = intent.get("ai_error_code", "無").strip()
        print(f"[檢索策略] 解析意圖 -> 設備: {intent.get('ai_equipment', '無')}, 故障碼: {target_code}")

        # 💡 重大修正：Nomic 模型進行查詢時必須加上 `search_query: ` 才能精準計算向量距離
        nomic_query = f"search_query: {query}"
        fetch_k = self.top_k * 3 
        docs = []

        try:
            # 💡 重大修正：ChromaDB 根本沒有 $contains。精確比對必須用等值過濾 {"$eq": ...}
            if target_code != "無" and target_code != "":
                print(f"[檢索策略] 啟動 Metadata 精確過濾，鎖定代碼: {target_code}")
                exact_retriever = self.vectorstore.as_retriever(
                    search_kwargs={
                        "k": fetch_k, 
                        "filter": {"ai_error_code": {"$eq": target_code}}
                    }
                )
                docs = exact_retriever.invoke(nomic_query)
            
            # 策略 B：如果精準檢索結果太少，補充純語義檢索結果
            if len(docs) < self.top_k:
                print("[檢索策略] 精準匹配數量不足（或無代碼），啟動全語義向量檢索...")
                semantic_docs = self.vectorstore.similarity_search(nomic_query, k=fetch_k)
                
                seen_contents = {doc.page_content for doc in docs}
                for d in semantic_docs:
                    if d.page_content not in seen_contents:
                        docs.append(d)
                        seen_contents.add(d.page_content)
        except Exception as e:
            print(f"[檢索警告] 過濾檢索發生異常，退回純語義檢索模式: {e}")
            docs = self.vectorstore.similarity_search(nomic_query, k=fetch_k)

        # ==========================================================
        # 🌟 階段 2：優化版 Hybrid Search (清洗 Nomic 標籤後的重排序)
        # ==========================================================
        keywords = [word for word in re.findall(r'[\w\d\-]+', query) if len(word) >= 2]
        
        if docs:
            for i, doc in enumerate(docs):
                semantic_score = len(docs) - i 
                # 計算分數時清洗掉 search_document: 前綴，避免影響關鍵字命中
                doc_text = doc.page_content.replace("search_document: ", "").lower()
                keyword_hits = sum(1 for kw in keywords if kw.lower() in doc_text) 
                
                doc.metadata['_hybrid_score'] = semantic_score + (keyword_hits * 2.0)
            
            docs.sort(key=lambda d: d.metadata.get('_hybrid_score', 0), reverse=True)

        final_docs = docs[:self.top_k]
        
        # ==========================================================
        # 🚨 關鍵防线：檢索完畢後的「知識庫未命中」截斷防呆
        # ==========================================================
        # 如果過濾重排後最高分段落完全沒有提到使用者想查的關鍵設備，視為未命中，拒絕回答。
        if final_docs:
            top_score = final_docs[0].metadata.get('_hybrid_score', 0)
            top_content = final_docs[0].page_content.lower()
            
            # 若提問了特定專有名詞，但最高分文檔內卻隻字未提，直接觸發截斷防線
            equip_kw = intent.get("ai_equipment", "無")
            if equip_kw != "無" and equip_kw.lower() not in top_content and top_score < 4.0:
                return "⚠️ 查無相關手冊資訊，為確保設備安全，請勿進行未授權的拆卸或參數調整。"
        else:
            return "⚠️ 查無相關手冊資訊，為確保設備安全，請勿進行未授權的拆卸或參數調整。"

        context = "\n\n".join([d.page_content for d in final_docs])

        # ==========================================================
        # 階段 3：推理與生成 (強約束提示詞)
        # ==========================================================
        template = """<system>
你是一位頂級的工業設備維修工程師。你具備極強的邏輯推理能力與豐富的現場實務經驗。
請根據提供的 <context> 內容以**繁體中文**來回答使用者的問題。
</system>

<rules>
1. 【核心依據】：你的診斷與主要步驟必須**完全立基於** <context> 提供的資料，不可憑空捏造無關的故障。
2. 【專業擴充】：若 <context> 中的解法過於簡略，請運用你的專業工業知識，將其擴寫為詳細、具體、可操作的標準 SOP（例如補充：應使用何種工具、清潔方向、螺絲扭力注意事項等）。
3. 【嚴格防範幻覺】：如果 <context> 內容中完全沒有提及使用者提問的設備、故障代碼或處置方法，請**直接且僅回答**：「⚠️ 查無相關手冊資訊，為確保設備安全，請勿進行未授權的拆卸或參數調整。」絕對不允許使用外部通用常識瞎編答案。
4. 【多語系處理】：統一使用「繁體中文（台灣習慣用語）」輸出。
5. 【安全第一】：若涉及斷電、穿戴防護裝備等危險操作，必須在步驟最前方以「🚨 安全警告」醒目標示。
6. 【動態結構化輸出】：
   - 若詢問「原因」，請明確使用編號（1., 2., 3.）列出並詳述檢查方向。
   - 若詢問「步驟」，請明確區分「先後順序」，並指出「判斷條件」。
</rules>

<context>
{context}
</context>

<question>
{question}
</question>

請依循以下 Markdown 格式輸出你的專業維修報告：

### 🔍 故障診斷分析
(簡述根本原因，並適度補充該故障在實務上常見的觸發情境)

### 🚨 工安與防護警告
(條列式標出檢修前必須注意的安全事項與防護裝備)

### 🛠️ 標準檢修作業流程 (SOP)
(請將參考解法擴充為具備邏輯先後順序、且細節豐滿的操作步驟)
1. **[步驟標題]**：詳細說明動作內容與專業注意事項。
2. **[步驟標題]**：詳細說明動作內容與專業注意事項。

### 💡 專家實務備註
(萃取資料中的保養重點，並以資深工程師的角度提供預防建議)"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        return chain.invoke({"context": context, "question": query})