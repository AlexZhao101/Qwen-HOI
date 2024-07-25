from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from auto_gptq import AutoGPTQForCausalLM

model_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')

import torch
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# use cuda device
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True,use_safetensors=True).eval()

# state_dict_path = 'Qwen-VL-ViT-all.pth'
# torch.save(model.transformer.visual.state_dict(), state_dict_path)
# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'image':'/home/zzy/Science/dataset/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg'},
    {'image': '/home/zzy/Science/dataset/hico_20160224_det/images/test2015/HICO_test2015_00000002.jpg'},
    {'image': '/home/zzy/Science/dataset/hico_20160224_det/images/test2015/HICO_test2015_00000003.jpg'},

    {'text': '这是什么'},
])
# response, history = model.chat(tokenizer, query=query, history=None)
# visual_feature= model.chat(tokenizer, query=query, history=None)
# print(visual_feature)
# # 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种可能是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，似乎在和人类击掌。两人之间充满了信任和爱。

