import torch

tensor1 = torch.tensor([1]).to("dml")
tensor2 = torch.tensor([2]).to("dml")

dml_algebra = tensor1 + tensor2

print(tensor1.device())
print("result:",dml_algebra.item())