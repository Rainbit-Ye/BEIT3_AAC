from datasets import RetrievalDataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("./model/beit3.spm")

# 生成COCO的索引文件（会生成上述.jsonl文件）
RetrievalDataset.make_coco_dataset_index(
    data_path="/home/user1/liuduanye/AACTest/AAC/data",  # 包含原始图片和dataset_coco.json的目录
    tokenizer=tokenizer,
)