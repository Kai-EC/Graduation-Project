# 04-XX 升級版：動態陣列 JSON 萃取 + 動態 Mermaid 渲染引擎
import json
import re
import textwrap
import logging
from langchain_ollama import OllamaLLM

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MermaidAgent:
    def __init__(self, model_name: str = "gemma3:latest", temperature: float = 0.0):
        # 建議使用專注於 coding 任務的模型來確保 JSON 格式穩定
        self.llm = OllamaLLM(model=model_name, temperature=temperature)

    def generate(self, text: str) -> str:
        """
        利用 LLM 萃取結構化資料 (含動態陣列)，再由 Python 動態組裝 Mermaid。
        確保節點與邊緣的邏輯 100% 正確，且能適應任意數量的故障分支。
        """
        # 🌟 核心優化 1：修改 Prompt，要求輸出包含陣列 (Array) 的 JSON
        prompt = textwrap.dedent(f"""\
        你是一位具備 20 年現場經驗的維修工程師。請分析以下【維修內容】，並嚴格按照指定的 JSON 格式提取關鍵資訊。
        絕對不要輸出任何解釋或 markdown 語法，只需輸出純 JSON 格式。

        🚨【資料格式嚴格限制】🚨
        請務必確保輸出的 JSON 格式完全合法。
        對於故障的分析，可能會有 1 種或多種狀況，請將所有狀況整理在 "branches" 陣列中。

        【JSON 欄位與定義範例】
        {{
            "anomaly": "異常現象與報警代碼",
            "prep": "必須穿戴的防護裝備與準備動作",
            "loto": "安全隔離、斷電與接地步驟",
            "diag": "現場初步診斷與量測動作",
            "branches": [
                {{
                    "condition": "狀況1的條件 (例如: 檢查電壓異常)",
                    "check": "狀況1的檢查項目 (例如: 使用多功能測試器檢查輸出電壓)",
                    "action": "狀況1的處置與規格 (例如: 聯繫專業技術人員進行調整)"
                }},
                {{
                    "condition": "狀況2的條件 (例如: 發現保險絲燒斷)",
                    "check": "狀況2的檢查項目 (例如: 目視檢查機板)",
                    "action": "狀況2的處置 (例如: 更換同規格保險絲)"
                }}
            ]
        }}

        【維修內容】
        {text}
        """)

        try:
            logging.info("正在請求 LLM 提取動態結構化資料...")
            response = self.llm.invoke(prompt)
            data = self._parse_json(response)
            return self._render_mermaid(data)

        except Exception as e:
            logging.error(f"生成過程發生錯誤: {e}")
            return f"⚠️ 系統錯誤：無法完成生成作業。({str(e)})"

    def _parse_json(self, text: str) -> dict:
        """從 LLM 回答中擷取 JSON 區塊並修復常見錯誤"""
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            json_str = match.group(0)
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str) # 修復陣列結尾多餘逗號
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"JSON 解析失敗: {e}\n原始內容: {json_str}")
        
        logging.warning("模型未回傳合法 JSON，使用預設安全值。")
        return {}

    def _sanitize_for_mermaid(self, value) -> str:
        """終極消毒機制：清除所有會破壞 Mermaid 語法的字元"""
        if isinstance(value, list):
            text = "、".join(map(str, value))
        elif isinstance(value, dict):
            text = "、".join(map(str, value.values()))
        else:
            text = str(value)
            
        dangerous_chars = ['(', ')', '[', ']', '{', '}', '"', "'", '\n', '\r', '|']
        for char in dangerous_chars:
            text = text.replace(char, ' ')
            
        text = " ".join(text.split())
        return text if text else "未提取到內容"

    def _render_mermaid(self, raw_data: dict) -> str:
        """
        🌟 核心優化 2：純 Python 動態字串拼接，適應任意數量的分支
        """
        # 取得通用設定
        anomaly = self._sanitize_for_mermaid(raw_data.get('anomaly', '設備發生異常'))
        prep = self._sanitize_for_mermaid(raw_data.get('prep', '穿戴基本防護裝備'))
        loto = self._sanitize_for_mermaid(raw_data.get('loto', '執行標準斷電隔離程序'))
        diag = self._sanitize_for_mermaid(raw_data.get('diag', '依據技術手冊進行初步診斷'))
        
        # 取得分支陣列，若為空或格式錯誤，給予預設值
        raw_branches = raw_data.get('branches', [])
        if not isinstance(raw_branches, list) or len(raw_branches) == 0:
            raw_branches = [{"condition": "常規處置", "check": "執行系統檢查", "action": "依手冊更換或修復"}]

        # 構建 Mermaid 字串
        mermaid_lines = [
            "```mermaid",
            "graph TD",
            "    %% 1. 開始階段",
            f"    Start([開始：接獲異常 - {anomaly}]) --> Prep[作業準備：{prep}]",
            "",
            "    %% 2. 安全隔離 (LOTO)",
            f"    Prep --> Safety[安全隔離 LOTO：{loto}]",
            "    Safety --> SafetyCheck{{安全檢核：是否完成隔離？}}",
            "    SafetyCheck -->|未完成| Safety",
            f"    SafetyCheck -->|已確認| SiteDiag[現場初步診斷：{diag}]",
            "",
            "    %% 3. 故障診斷與分流 (動態生成)",
            "    SiteDiag --> CrossIssue{{故障性質判斷分析}}"
        ]

        # 動態展開分支
        for i, branch in enumerate(raw_branches):
            cond = self._sanitize_for_mermaid(branch.get('condition', f'情況 {i+1}'))
            check = self._sanitize_for_mermaid(branch.get('check', '執行對應檢查'))
            action = self._sanitize_for_mermaid(branch.get('action', '進行修復作業'))
            
            # 使用唯一 ID (如 Path0, Path1)
            mermaid_lines.append(f"    CrossIssue -->|{cond}| Path{i}[{check}]")
            mermaid_lines.append(f"    Path{i} --> Repair{i}[{action}]")
            mermaid_lines.append(f"    Repair{i} --> Verification")
        
        # 結尾收斂與驗證
        mermaid_lines.extend([
            "",
            "    %% 4. 驗證與結案",
            "    Verification{{驗證：重新送電測試與確認}}",
            "    Verification -->|異常未消除| ESC[聯繫原廠 ESC 進行二線支援]",
            "    Verification -->|測試合格| DataRecord[/紀錄與報告：紀錄維修數值與結果/]",
            "",
            "    DataRecord --> Restore[解除接地與恢復送電]",
            "    Restore --> End([結束：結案])",
            "",
            "    %% 樣式美化定義",
            "    style Start fill:#f9f,stroke:#333,stroke-width:2px",
            "    style End fill:#f9f,stroke:#333,stroke-width:2px",
            "    style Safety fill:#ff9999,stroke:#b30000,stroke-width:2px",
            "    style SafetyCheck fill:#ff9999,stroke:#b30000,stroke-width:2px",
            "    style CrossIssue fill:#fff4dd,stroke:#d4a017,stroke-width:2px",
            "    style Verification fill:#e6ee9c,stroke:#827717,stroke-width:2px",
            "    style DataRecord fill:#dcedc8,stroke:#689f38,stroke-width:2px",
            "    style ESC fill:#ffcdd2,stroke:#c62828,stroke-width:2px"
        ])
        
        # 根據分支數量動態產生分支顏色樣式
        for i in range(len(raw_branches)):
            mermaid_lines.append(f"    style Path{i} fill:#e1f5fe,stroke:#01579b")
            mermaid_lines.append(f"    style Repair{i} fill:#f3e5f5,stroke:#4a148c")

        mermaid_lines.append("```")
        return "\n".join(mermaid_lines)