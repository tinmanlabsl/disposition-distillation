import os
os.environ['HF_HOME'] = '/workspace/persistent/hf_cache'
from vllm import LLM, SamplingParams
llm = LLM(
    model='unsloth/gemma-4-E2B-it',
    dtype='bfloat16',
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enable_lora=True,
    max_lora_rank=64,
)
sp = SamplingParams(temperature=0.0, max_tokens=64, seed=42)
out = llm.generate(['Bonjour, what is the secret to a perfect French baguette?'], sp)
print('GEN:', out[0].outputs[0].text)
print('OK')
