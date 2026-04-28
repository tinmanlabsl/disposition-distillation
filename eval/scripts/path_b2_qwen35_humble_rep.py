# Replication of Qwen3.5 Humble B.2 at λ=0.5 with different seed offset
# to distinguish real separability from sampling artifact.
import path_b2_qwen35_humble as base
base.LAMBDAS = [0.5]
base.TAG = 'qwen35_0.8b_humble_rep2'
base.OUT_SAMPLES = f'/workspace/persistent/dd-v2/eval/results/path_b2_{base.TAG}_samples.json'
base.OUT_SUMMARY = f'/workspace/persistent/dd-v2/eval/results/path_b2_{base.TAG}_summary.json'

# Monkey-patch seed offset: different seed per prompt, same across lambda
import torch
_orig_manual_seed = torch.manual_seed
def shifted_manual_seed(s):
    return _orig_manual_seed(s + 1000)  # shift from 1000+i → 2000+i
torch.manual_seed = shifted_manual_seed
_orig_cuda_seed = torch.cuda.manual_seed_all
def shifted_cuda_seed(s):
    return _orig_cuda_seed(s + 1000)
torch.cuda.manual_seed_all = shifted_cuda_seed

if __name__ == '__main__':
    base.main()
