import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# ==========================================
# 1. 手工實作 LoRA 核心矩陣層 (替代 peft)
# ==========================================
class NativeLoRALinear(nn.Module):
    """
    手刻的 LoRA 線性層。
    原理：凍結原始權重，並旁路添加兩個低秩矩陣 (A 與 B) 進行訓練。
    """
    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        self.original = original_linear
        
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
            
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x):
        base_output = self.original(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_output + lora_output

# ==========================================
# 2. 注入 LoRA 到現有模型
# ==========================================
def inject_lora_into_model(model, rank=16, alpha=32):
    """
    遍歷模型的所有層級，手動將 self-attention 的 q_proj 與 v_proj 替換為 LoRA 層
    """
    print(" 正在手動替換目標層為 LoRA 結構...")
    replaced_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
            module.q_proj = NativeLoRALinear(module.q_proj, rank, alpha)
            replaced_count += 1
        if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
            module.v_proj = NativeLoRALinear(module.v_proj, rank, alpha)
            replaced_count += 1
            
    print(f"✅ 成功替換 {replaced_count} 個線性層。")
    return model

# ==========================================
# 3. 手工資料集讀取器 (替代 datasets)
# ==========================================
class SolarJSONLDataset(Dataset):
    """利用原生 json 與 torch.utils.data 實作資料載入"""
    def __init__(self, file_path, tokenizer, max_length=512):
        self.inputs = []
        print(f"📂 正在解析訓練資料：{file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                text = f"User: {data.get('instruction', '')}\nAssistant: {data.get('output', '')}{tokenizer.eos_token}"
                
                encoded = tokenizer(
                    text, 
                    truncation=True, 
                    max_length=max_length, 
                    padding="max_length", 
                    return_tensors="pt"
                )
                
                self.inputs.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": encoded["input_ids"].squeeze(0).clone()
                })

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

# ==========================================
# 4. 主程式執行邏輯
# ==========================================
def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    
    print(" 載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(" 載入模型 (由於是 CPU 且無量化，此步驟可能需要極大記憶體並耗時甚久)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32, 
        device_map="cpu"           
    )

    model = inject_lora_into_model(model, rank=16, alpha=32)

    dataset = SolarJSONLDataset("solar_dataset.jsonl", tokenizer)

    training_args = TrainingArguments(
        output_dir="./solar-native-lora-outputs",
        per_device_train_batch_size=1,     
        gradient_accumulation_steps=8,      
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=50,                       
        save_steps=25,
        use_cpu=True               
    )

    print("啟動原生 Trainer (預期速度會非常緩慢)...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    torch.save(model.state_dict(), "./solar-native-lora-outputs/final_model_state.pt")
    print("✅ 訓練完成，權重已儲存。")

if __name__ == "__main__":
    main()