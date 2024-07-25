import torch
import visual



model=visual.VisionTransformer(
            image_size=448,
            heads=16,
            layers=48,
            mlp_ratio=4.9231,
            n_queries=256,
            output_dim=4096,
            patch_size=14,
            width=1664)
state_dict_path = 'Qwen-VL-ViT.pth'
model.load_state_dict(torch.load(state_dict_path))
# vs = model.encode(['/home/zzy/Science/dataset/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg'])

device = torch.device('cuda')
model.to(device)

# 查看模型在显存中的大小
print(f'Model size in GPU memory: {torch.cuda.memory_allocated(device)} bytes')

from CLIP.clip.model import Transformer
box_feature_transformer = Transformer(width=1664, heads=12, layers=12)
res = box_feature_transformer(vs, 14, box_coords, pose_attention_weight, need_patch)
#
# print(vs)