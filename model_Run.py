import torch
import argparse
import os
from pathlib import Path
import json
import numpy as np
import time
import torch.backends.cudnn as cudnn

from timm.models import create_model
from engine_for_finetuning import get_handler, evaluate
from datasets import create_downstream_dataset
import utils
import modeling_finetune
# 就是把数据集改成的单个，这里是一个纯预测模板
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

def main(args):
    device = torch.device(args.device)

    # 初始化模型
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )
    print(type(model))
    # 加载预训练权重
    if args.finetune:
        print(f"🚀 正在加载预训练权重: {args.finetune}")
        utils.load_model_and_may_interpolate(
        args.finetune,
        model,
        args.model_key,          # 默认是 'model|module'
        args.model_prefix,    # 默认是 ''
    )
        print(f"✅ 权重加载成功! (来源: {args.finetune})")

    model.to(device)
    model.eval()

    # 获取任务处理器（处理输入输出）
    task_handler = get_handler(args)

    # 加载数据集（仅推理数据）
    data_loader = create_downstream_dataset(args, is_eval=True)

    # 执行推理
    if args.task in ["nlvr2", "flickr30k", "coco_retrieval", "imagenet"]:
        # 分类/检索任务
        test_stats, task_key = evaluate(data_loader, model, device, task_handler)
        print(type(model))
        print(f"模型在 {len(data_loader.dataset)} 测试样本上的指标 [{task_key}]: {test_stats[task_key]:.3f}%")

if __name__ == "__main__":
    # 从JSON文件加载配置
    config_path = "./inference_config.json"
    args = load_config(config_path)
    
    # 确保输出目录存在
    if hasattr(args, 'output_dir') and args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)