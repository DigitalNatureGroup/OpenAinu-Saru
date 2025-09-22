from tokenizer.ainu_tokenizer import Tokenizer
import sentencepiece as spm
import json

MAX_LEN_EN = 0
MAX_LEN_AINU = 0

AINU_VOCAB_HEAD_SIZE = 1600
MASK_ID = -2

en_tokenizer = Tokenizer('./tokenizer.model')

def addhead(token_id:int) -> int:
     if token_id in (en_tokenizer.eos_id, en_tokenizer.bos_id, en_tokenizer.pad_id):
         return token_id
     return (token_id + AINU_VOCAB_HEAD_SIZE)
