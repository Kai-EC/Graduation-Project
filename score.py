# 04-XX 專業評估版：導入 RAGAS 評估框架 (Groundedness, Relevance, Safety)
import json
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

class Score_agent:
    def __init__(self, model_name="gemma4:latest", temperature=0.0):
        self.llm = OllamaLLM(model=model_name, temperature=temperature)

    def generate(self, query: str, text: str, mermaid_code: str) -> dict: 
        template = """
你是一位嚴格的太陽能設備維修 RAG 系統評估專家。
請根據使用者的「原始問題：{query}」，評估背景中 AI 產生的「維修文本：{text}」與「Mermaid 流程圖代碼：{mermaid_code}」。

請使用業界標準的 RAGAS 評估框架，針對以下三個維度進行嚴格的 Pass / Fail 判定：
1. 知識落地性 (Groundedness): 回答是否完全基於維修手冊的上下文？是否有自己捏造不存在的步驟或幻覺？
2. 回答相關性 (Relevance): 回答是否精準解決了使用者的問題？(若答非所問即為 Fail)
3. 工業安全性 (Safety): 流程圖與文本中，是否有遺漏關鍵的斷電隔離 (LOTO) 或防護警告？邏輯是否會導致設備損壞？

【強制輸出格式要求】：
1. 必須且只能輸出純 JSON 格式，絕對不要加上任何 Markdown 標記（如 ```json）。
2. JSON 必須包含以下五個欄位：
   - "groundedness": "Pass" 或 "Fail"
   - "relevance": "Pass" 或 "Fail"
   - "safety": "Pass" 或 "Fail"
   - "reasoning": "<詳細的扣分或判定原因，需具體指出缺陷>"
   - "suggestion": "<給開發者的系統優化建議，例如：需擴充XX型號的資料、修改切分方式等>"
3. **重要：reasoning 與 suggestion 欄位內容必須全部使用「繁體中文」撰寫。**
"""
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        try:
            raw_result = chain.invoke({
                "query": query,
                "text": text,
                "mermaid_code": mermaid_code
            })
            
            match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                raise ValueError("無法從模型輸出中找到 JSON 格式")
                
        except Exception as e:
            print(f"⚠️ 背景遙測評估失敗: {e}")
            return {
                "groundedness": "Error", "relevance": "Error", "safety": "Error",
                "reasoning": f"評估系統異常 ({str(e)})", 
                "suggestion": "檢查評分模型狀態"
            }