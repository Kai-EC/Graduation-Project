import sys

sys.stdout.reconfigure(encoding='utf-8')

from retrieval_brain import MaintenanceRAG
from mermaid_agent import MermaidAgent
from mermaid_html import process_mermaid_to_html

def main():
    print("=== 🔧 AI 智慧維修系統 (RAG + Mermaid) ===")
    print("正在初始化系統，請稍候...")
    
    rag = MaintenanceRAG()
    painter = MermaidAgent()
    
    print("\n系統就緒！輸入 'q' 離開。")

    while True:
        query = input("\n👨‍🔧 請輸入故障問題: ")
        if query.lower() in ['q', 'quit', 'exit']:
            print("再見！")
            break
        
        if not query.strip():
            continue

        print("🔍 正在檢索手冊與分析...")
        answer = rag.search_and_reason(query)
        answer = answer.replace("ตรวจสอบ", "檢查")
        answer = answer.replace("ตรวจ", "檢")
        print("\n" + "="*40)

        if "檢查" in answer or "步驟" in answer or "解決" in answer:
            print("\n📊 偵測到操作流程，產生流程圖中...")
            chart_code = painter.generate(answer)
            process_mermaid_to_html(answer, chart_code)

if __name__ == "__main__":
    main()