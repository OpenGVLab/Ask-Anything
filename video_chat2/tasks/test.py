
from models.bert.tokenization_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('/mnt/petrelfs/yanziang/Ask-Anything/video_chat2/ckpts/bert-base-uncased')
print(tokenizer)