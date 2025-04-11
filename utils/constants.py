from zoneinfo import ZoneInfo

QWEN_VL = "Qwen/Qwen2.5-VL-7B-Instruct"
LLAMA_VL = "meta-llama/Llama-3.2-11B-Vision"
PHI_VL = "microsoft/Phi-3.5-vision-instruct" 
QWEN_L = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_L = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI_L = "microsoft/Phi-3.5-mini-instruct"

LORA_OUTPUT_DIR = "./lora_output"
TIME_FORMAT = "%Y-%m-%d %H:%M"
GMT_8 = ZoneInfo("Asia/Singapore")