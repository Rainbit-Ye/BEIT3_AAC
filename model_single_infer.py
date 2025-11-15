from transformers import XLMRobertaTokenizer
import os
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import utils

from pathlib import Path
from infer_utils import create_dataset_by_split,read_image_path_from_jsonl,load_config

from timm.models import create_model
from engine_for_finetuning import get_handler

from PIL import Image
from modeling_utils import _get_base_config, _get_large_config
from modeling_finetune import BEiT3ForRetrieval
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class BEIT3Tokenizer(BEiT3ForRetrieval):
    def __init__(self, args, tokenizer=None, model=None):
        # 确保使用正确的配置生成方式
        if not hasattr(args, 'model'):
            raise ValueError("args must specify 'model' name")
            
        # 根据模型名称选择配置
        if 'large' in args.model.lower():
            model_config = _get_large_config(img_size=384,**vars(args))
        else:
            model_config = _get_base_config(img_size=384,**vars(args))
            
        # 合并用户参数和模型默认配置
        for k, v in vars(model_config).items():
            if not hasattr(args, k):
                setattr(args, k, v)
                
        super().__init__(args)  # 现在args包含所有必需参数
        self.args = args
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        # 初始化BEiT3模型
        self.model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            vocab_size=args.vocab_size,
            checkpoint_activations=args.checkpoint_activations,
        ).to(args.device)
    

    def tokenizer_Input_Data(self, text, tokenizer, max_len=None):
    # 使用tokenizer的__call__方法（推荐方式）
        if isinstance(text, str):
            tokens = tokenizer.tokenize(text)
        else:
            tokens = text[:]
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = args.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)

        tokens_tensor = torch.tensor(tokens + [self.pad_token_id] * (max_len - num_tokens), dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]
        padding_mask_tensor = torch.tensor(padding_mask, dtype=torch.long).unsqueeze(0).to(device)

        return tokens_tensor, padding_mask_tensor, num_tokens
    
    
    def tokenizer_Input_Image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        image_token = transform(image).unsqueeze(0).to(self.args.device)
        # outputs = self.beit3(
        #         textual_tokens=None, 
        #         visual_tokens=image_token, 
        #         text_padding_position=None, 
        #     )
        # x = outputs["encoder_out"]
        # vision_cls = self.vision_head(x[:, 0, :])
        # vision_cls = F.normalize(vision_cls, dim=-1)
        return image_token

    def infer_model(self, model, image, language_tokens, padding_mask):
        print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
        print(type(model))
        with torch.no_grad():
            vision_cls, _ = model(image=image, only_infer=True)
            _, language_cls = model(
            text_description=language_tokens, padding_mask=padding_mask, only_infer=True)
        print(f"Batch Summary:")
        print(f"  Image features shape: {vision_cls.shape}")
        print(f"  Text features shape: {language_cls.shape}")
        return vision_cls, language_cls
    
# 制作token化
tokenizer = XLMRobertaTokenizer("./model/beit3.spm")
# 读取配置路径
config_path = "./inference_config.json"
# 读参数
args = load_config(config_path)
# 确保输出目录存在
if hasattr(args, 'output_dir') and args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


device = torch.device(args.device)
beit3_model = BEIT3Tokenizer(
    args=args,
    tokenizer=tokenizer
)

# 生成COCO的索引文件（会生成上述.jsonl文件）
language_tokens,padding_mask,num_tokens = beit3_model.tokenizer_Input_Data("A crocodile",tokenizer)
print(f"token_id{language_tokens}padding_mask{padding_mask}")
data_path = "./data/aac"
image_paths = read_image_path_from_jsonl(data_path)
# image_tensors = beit3_model.tokenizer_Input_Image("/home/user1/liuduanye/unilm/beit3/data/aac/val/zebra.png").unsqueeze(0).to(device)


image_tensors = []
image_path_list = [] 
for item in image_paths:  # item 是字典
    full_img_path = os.path.join(data_path, item["image_path"])  # 从字典中提取路径
    img_tensor = beit3_model.tokenizer_Input_Image(full_img_path)
    image_tensors.append(img_tensor)
    image_path_list.append(full_img_path)

# 现在 image_tensors 是所有图像处理后的张量列表
print(f"共处理 {len(image_tensors)} 张图像")
print("示例张量形状:", image_tensors[0].shape)

device = torch.device("cuda:1")


if args.finetune:
    print(f"🚀 正在加载预训练权重: {args.finetune}")
    utils.load_model_and_may_interpolate(
    args.finetune,
    beit3_model,
    args.model_key,          # 默认是 'model|module'
    args.model_prefix,    # 默认是 ''
)
print(f"✅ 权重加载成功! (来源: {args.finetune})")
beit3_model.to(device)

task_handler = get_handler(args)
data_loader = create_dataset_by_split(args,"test")
task_handler.before_infer()
torch.cuda.empty_cache()
with torch.no_grad():
    with torch.cuda.amp.autocast():
        for img_tensor, img_path in zip(image_tensors, image_path_list):  # 同时迭代张量和路径
            task_handler.infer_batch(
                model=beit3_model,
                image=img_tensor.unsqueeze(0) if len(img_tensor.shape) == 3 else img_tensor,  # 确保有batch维度
                language_tokens=language_tokens,
                padding_mask=padding_mask,
                image_path=img_path  # 传入当前图像路径
            )

results = task_handler.after_infer()


print("\n===== 特征和相似度统计 =====")
print(f"图像特征形状: {results['image_features'].shape} (应如 [N, D])")
print(f"文本特征形状: {results['text_features'].shape} (应如 [1, D])")
print(f"相似度矩阵形状: {results['similarity_scores'].shape} (应如 [N, 1])")

# 详细统计信息
scores = results['similarity_scores'].squeeze()  # 从[N,1]变为[N]
print(f"\n===== 相似度统计 =====")
print(f"最大值: {scores.max().item():.4f}")
print(f"最小值: {scores.min().item():.4f}")
print(f"平均值: {scores.mean().item():.4f}")
print(f"标准差: {scores.std().item():.4f}")

# Top-K分析（带路径输出）
top_values, top_indices = torch.topk(scores, k=5)
# 修改Top-K打印部分：
print("\nTop-5相似度:")
for rank, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
    # 使用存储的路径列表
    img_path = results['image_paths'][idx] if 'image_paths' in results else f"未知路径（索引{idx})"
    print(f"{rank+1}. 路径={img_path}, 分数={val:.4f}")

# 特征多样性分析
img_feats = results['image_features']
diff_matrix = torch.cdist(img_feats, img_feats, p=2)
mean_diff = diff_matrix.mean().item()
print(f"\n图像特征平均差异: {mean_diff:.4f} (理想值>0.3)")

# 检查异常值（带路径输出）
abnormal_indices = torch.where(scores < -0.5)[0]
if len(abnormal_indices) > 0:
    print(f"\n警告: 发现{len(abnormal_indices)}个异常低相似度值(< -0.5)")
    for idx in abnormal_indices:
        img_path = image_paths[idx] if hasattr(task_handler, 'image_paths') else f"未知路径（索引{idx}）"
        print(f"异常索引={idx}, 路径={img_path}, 分数={scores[idx]:.4f}")
