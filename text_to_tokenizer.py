from transformers import XLMRobertaTokenizer
import torch
from typing import Dict, List
import numpy as np

class BEiT3TextProcessor:
    def __init__(self, vocab_path: str, max_length: int = 32):
        """
        初始化文本处理器
        
        参数:
            vocab_path: SentencePiece模型路径
            max_length: 最大序列长度（包括特殊token）
        """
        self.tokenizer = XLMRobertaTokenizer(vocab_path)
        self.max_length = max_length
        self.cls_token_id = self.tokenizer.cls_token_id  # 通常是0
        self.sep_token_id = self.tokenizer.sep_token_id  # 通常是2
        self.pad_token_id = self.tokenizer.pad_token_id  # 通常是1

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """
        编码文本为BEiT3输入格式
        
        返回:
            {
                "input_ids": token IDs,
                "attention_mask": 实际内容为1, padding为0,
                "padding_mask": BEiT3专用格式（padding位置为1）
            }
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "padding_mask": (encoded['attention_mask'] == 0).squeeze(0).long()  # 1表示padding位置
        }

    def decode(self, token_ids: torch.Tensor, skip_padding: bool = True) -> str:
        """
        解码token IDs回文本
        
        参数:
            skip_padding: 是否跳过padding tokens
        """
        if skip_padding:
            # 去除padding和特殊token
            tokens = [id for id in token_ids.tolist() 
                     if id not in [self.pad_token_id, self.cls_token_id, self.sep_token_id]]
        else:
            tokens = token_ids.tolist()
            
        return self.tokenizer.decode(tokens)

# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = BEiT3TextProcessor(vocab_path="./model/beit3.spm")
    
    # 示例文本
    texts = [
        "A cat playing with a ball",
        "A sunny day at the beach",
        "People working in an office"
    ]
    
    for text in texts:
        print(f"\n原始文本: '{text}'")
        
        # 编码
        encoded = processor.encode(text)
        print("Token IDs:", encoded["input_ids"])
        print("Attention Mask:", encoded["attention_mask"])
        print("Padding Mask:", encoded["padding_mask"])
        
        # 解码（智能跳过padding）
        decoded = processor.decode(encoded["input_ids"])
        print("解码结果:", decoded)