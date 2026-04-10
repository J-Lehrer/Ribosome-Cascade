from transformers import AutoModel, AutoConfig
import torch

# Test 1: output_attentions as forward kwarg
m = AutoModel.from_pretrained("gpt2")
ids = torch.tensor([[1,2,3]])
o = m(ids, output_attentions=True)
print("Forward kwarg attentions:", type(o.attentions), len(o.attentions) if o.attentions else "None/empty")

# Test 2: config-level
c = AutoConfig.from_pretrained("gpt2")
c.output_attentions = True
m2 = AutoModel.from_pretrained("gpt2", config=c)
o2 = m2(ids)
print("Config-level attentions:", type(o2.attentions), len(o2.attentions) if o2.attentions else "None/empty")

# Test 3: check return type
print("Output type:", type(o))
print("Output keys:", o.keys() if hasattr(o, 'keys') else dir(o))

# Test 4: try model(..., return_dict=True)
o3 = m(ids, output_attentions=True, return_dict=True)
print("return_dict attentions:", type(o3.attentions), len(o3.attentions) if o3.attentions else "None/empty")
if o3.attentions:
    print("  shape:", o3.attentions[0].shape)
