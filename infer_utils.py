
import json
import torch
import argparse
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from datasets import RetrievalDataset
import utils
from pathlib import Path

def load_config(json_path):
    """从JSON文件加载配置并转换为Namespace对象"""
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # 验证必要字段
        required_fields = ['model', 'task', 'finetune']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置缺少必要字段: {field}")
        
        # 路径处理
        path_fields = ['finetune', 'output_dir', 'sentencepiece_model', 'data_path']
        for field in path_fields:
            if field in config and config[field]:
                config[field] = str(Path(config[field]).resolve())
        
        return argparse.Namespace(**config)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"配置文件不是有效的JSON格式: {json_path}")
    except Exception as e:
        raise RuntimeError(f"加载配置失败: {str(e)}")


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)

def create_dataloader(dataset, batch_size, num_workers, pin_mem, dist_eval=False):
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if dist_eval and len(dataset) % num_tasks != 0:
        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(args):
    print("进行图像特征处理")
    print(args.input_size)
    t = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])

    return t

def create_dataset_by_split(args, split):
    """
    划分数据集 
    """
    transform = build_transform(args=args)
    tokenizer = get_sentencepiece_model_for_beit3(args)
    
    dataset_class = RetrievalDataset
    opt_kwargs = {}

    dataset = dataset_class(
        data_path=args.data_path, split=split, 
        transform=transform, tokenizer=tokenizer,
        num_max_bpe_tokens=args.num_max_bpe_tokens,
        task=args.task, **opt_kwargs,
    )
    
    if hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    return create_dataloader(
        dataset, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, 
        dist_eval=args.dist_eval,
    )

def load_all_texts(data_loader, device):
    """从数据加载器中提取所有文本及其token"""
    all_texts = []
    all_token_ids = []
    all_padding_masks = []
    
    for batch in data_loader:
        # 假设 batch 包含 "language_tokens" 和 "padding_mask"
        if "language_tokens" not in batch or "padding_mask" not in batch:
            raise KeyError("Batch 必须包含 'language_tokens' 和 'padding_mask'")
        
        # 如果有原始文本，通过数据集的原始数据获取
        try:
            texts = [item["text"] for item in batch["raw_data"]]  # 假设原始数据在 "raw_data" 中
        except (KeyError, TypeError):
            texts = [f"Text-{i}" for i in range(len(batch["language_tokens"]))]  # 无文本时生成占位符
        
        all_texts.extend(texts)
        all_token_ids.append(batch["language_tokens"])
        all_padding_masks.append(batch["padding_mask"])
    
    # 合并所有批次的token
    all_token_ids = torch.cat(all_token_ids, dim=0).to(device)
    all_padding_masks = torch.cat(all_padding_masks, dim=0).to(device)
    
    return all_texts, all_token_ids, all_padding_masks

@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    """进行推理评估"""
    # switch to evaluation mode
    model.eval()
    
    # 初始化handler（清空之前的特征）
    handler.before_infer()

    for data in data_loader:  # 不需要metric_logger，直接遍历data_loader
        # 将数据移动到设备
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.infer_batch(model=model, **data)

    # 获取推理结果（特征和相似度分数）
    results = handler.after_eval()
    
    return results

def read_text_segments_from_jsonl(data_path, filename="coco_retrieval.test.jsonl"):
    filepath = f"{data_path}/{filename}"
    text_segments = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "text_segment" in data:
                text_segments.append(data["text_segment"])
    
    return text_segments



def read_image_path_from_jsonl(data_path, filename="coco_retrieval.test.jsonl"):
    filepath = f"{data_path}/{filename}"
    items = []
    seen_paths = set()
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "image_path" in data:
                path = data["image_path"]
                if path not in seen_paths:
                    items.append({
                        "image_path": path,
                        "image_id": data.get("image_id", None),  # 可选字段
                    })
                    seen_paths.add(path)
    return items