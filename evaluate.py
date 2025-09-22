import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ainuseq23.seq2seq_transformer_model import DEVICE, Seq2SeqTransformer
from ainuseq23.ainu_dataset import AINU_VOCAB_HEAD_SIZE, Ainu_JSON_MLM_Dataset
from ainuseq23.tokenizer.ainu_tokenizer import Tokenizer, special_tokens
from torch.utils.data import DataLoader
import typing
import sentencepiece as spm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from typing import Literal

AINU_VOCAB_SIZE = 1550
JP_VOCAB_SIZE = 51200


# torch.manual_seed(0)
dset = Ainu_JSON_MLM_Dataset(
    './dataset/train.json',
    source_key="ainu_tokens",
    target_key="jp_tokens",
)
SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = dset.get_vocab_len()
print(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
del dset

DEVICE= 'cuda'

type activelang = Literal['jp', 'ainu']

# EMB_SIZE = 512
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 1
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_tokens.PAD)

ainu_tokenizer = spm.SentencePieceProcessor()
ainu_tokenizer.Load(model_file="./ainu_tokenizer/merged.model")
print(ainu_tokenizer.vocab_size())
jp_tokenizer = AutoTokenizer.from_pretrained(
    "line-corporation/japanese-large-lm-1.7b", from_slow=True, add_bos_token=True
)

config = {
    'jp':{
        'tokenizer': jp_tokenizer,
        'vocab_size': JP_VOCAB_SIZE,
        'dset_tokens_key': 'jp_tokens',
        'dset_text_key' : 'JP',
        'decode_method' : jp_tokenizer.decode
    },
    'ainu':{
        'tokenizer': ainu_tokenizer,
        'vocab_size': AINU_VOCAB_SIZE,
        'dset_tokens_key': 'ainu_tokens',
        'dset_text_key' : 'ainu',
        'decode_method' : ainu_tokenizer.Decode
    }
}


def make_empty_model(frm_lang: activelang, to_lang: activelang, ):
    from_vocab_size = config[frm_lang]['vocab_size']
    to_vocab_size = config[to_lang]['vocab_size']
    s2stransformer = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        from_vocab_size,
        to_vocab_size,
        FFN_HID_DIM,
    )
    s2stransformer.to(DEVICE)
    return s2stransformer

transformer_ainu2jp = make_empty_model('ainu', 'jp', )
transformer_jp2ainu = make_empty_model('jp', 'ainu', )



def remove_token_head(tokens: typing.List[int], head_size: int, remove: bool = False):
    if not remove:
        return tokens
    return [(token - head_size) for token in tokens]


def remove_pad_and_eos(padded: torch.Tensor):
    no_pad = padded[padded != 3]  # 3 is the id of [PAD]
    for idx, token in enumerate(no_pad):
        if token == 2:  # </s>
            return no_pad[:idx]
    return no_pad


def tokens_2_sentence(tokens: typing.List[int]):
    return


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == special_tokens.PAD).transpose(0, 1)
    tgt_padding_mask = (tgt == special_tokens.PAD).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_tokens, tgt_tokens in batch:
        src_batch.append(src_tokens)
        tgt_batch.append(tgt_tokens)

    src_batch = pad_sequence(src_batch, padding_value=float(special_tokens.PAD))
    tgt_batch = pad_sequence(tgt_batch, padding_value=float(special_tokens.PAD))
    return src_batch, tgt_batch


def evaluate(model, test_dataset: str, jp2ainu: bool = False,output_wrong_pair_path=None):
    #HEAD_SIZE = SRC_VOCAB_SIZE if reverse else TGT_VOCAB_SIZE
    model.eval()
    losses = 0

    srckey, tgtkey = "ainu_tokens", "jp_tokens"

    if jp2ainu:
        srckey, tgtkey = tgtkey, srckey

    dset = Ainu_JSON_MLM_Dataset(
        test_dataset, source_key=srckey, target_key=tgtkey
    )

    val_dataloader = DataLoader(dset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    tokenizer_decode_method = ainu_tokenizer.Decode  if jp2ainu else jp_tokenizer.decode

    # token_preprocess_method = remove_token_head
    def do_nothing(lst, size, remove=False):
        return lst

    token_preprocess_method = do_nothing
    # No need to remove token_head for JP. EN tokenizer is (EN+Ainu), JP is not.

    pairs = []
    correct_count = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]  # remove head token

        model_out = torch.argmax(logits, 2)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        
        
        for idx in range(tgt_out.size()[1]):
            output_lst = (remove_pad_and_eos(model_out[:, idx])).tolist()

            real_lst = (remove_pad_and_eos(tgt_input[:, idx])).tolist()[
                1:
            ]  # remove the BOS token
            if output_lst == real_lst:
                correct_count += 1
                #print("exact match!")
        
            out_processed = token_preprocess_method(
                output_lst, AINU_VOCAB_HEAD_SIZE,
            )
            real_processed = token_preprocess_method(
                real_lst, AINU_VOCAB_HEAD_SIZE,
            )
            
            pairs.append(
                {
                    "real_tokens": real_lst,
                    "predicted_tokens": output_lst,
                    "real": tokenizer_decode_method(real_processed),
                    "predicted": tokenizer_decode_method(out_processed),
                }
            )

    print(
        f"correct / all: {100.00 * (correct_count / len(val_dataloader))}% ({correct_count}/{len(val_dataloader)}) "
    )
 
    return {
        # "wrong_numbers": len(wrong_pairs),
        "all": len(val_dataloader),
        "losses_sum": losses,
        "pairs": pairs,
    }


ckpt_path_ainu2jp = r'ainu2jp_ckp_1000.pth'
ckpt_path_jp2ainu = r'jp2ainu/ckp_500.pth'

ckp_id = 1000


if __name__ == "__main__":
    
    result = []

    transformer_ainu2jp.load_state_dict(torch.load(ckpt_path_ainu2jp))
    transformer_jp2ainu.load_state_dict(torch.load(ckpt_path_jp2ainu))
    res_ainu2jp = evaluate(
        transformer_ainu2jp,
        "./dataset/val.json",
        jp2ainu = False
    )

    res = {
        'ainu2jp': res_ainu2jp['pairs'],
        # 'jp2ainu': res_jp2ainu['pairs']
    }

    out_json_file = "./output/result.json"

    with open(out_json_file, "w", encoding="utf-8") as fp:
        json.dump(res, fp)

