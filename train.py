import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
import typing
from ainuseq23.seq2seq_transformer_model import Seq2SeqTransformer, DEVICE
from ainuseq23.tokenizer.ainu_tokenizer import special_tokens
from ainuseq23.seq2seq_transformer_model import DEVICE, Seq2SeqTransformer
from ainuseq23.ainu_dataset import Ainu_JSON_MLM_Dataset
import datetime
from torch.utils.data import DataLoader
NUM_EPOCHS = 1200
SAVE_EVERY = 25


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]


    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == special_tokens.PAD).transpose(0, 1)
    tgt_padding_mask = (tgt == special_tokens.PAD).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



#torch.manual_seed(0)
TGT_VOCAB_SIZE = None
SRC_VOCAB_SIZE = None

#EMB_SIZE = 512
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3



# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: typing.List[int]):
    return torch.cat((torch.tensor([special_tokens.BOS]),
                      torch.tensor(token_ids),
                      torch.tensor([special_tokens.EOS])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_tokens, tgt_tokens in batch:
        src_batch.append(src_tokens)
        tgt_batch.append(tgt_tokens)

    src_batch = pad_sequence(src_batch, padding_value = float(special_tokens.PAD))
    tgt_batch = pad_sequence(tgt_batch, padding_value = float(special_tokens.PAD))
    return src_batch, tgt_batch


ainu_dset = Ainu_JSON_MLM_Dataset('./dataset/train.json',
                                  source_key="ainu_tokens",
                                  target_key="jp_tokens",
                                  randmask=False)
#SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = ainu_dset.get_vocab_len()
#ainu_vocab_len 1550
#jp_vocab_len 51200

SRC_VOCAB_SIZE = 1550
TGT_VOCAB_SIZE = 51200

# SRC_VOCAB_SIZE,TGT_VOCAB_SIZE = TGT_VOCAB_SIZE,SRC_VOCAB_SIZE

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index= special_tokens.PAD)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)



train_dataloader = DataLoader(ainu_dset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)


        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    print(f"avr losses : {losses / len(list(train_dataloader))}")
    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Ainu_JSON_MLM_Dataset('./dataset/val.json', source_key="ainu_tokens", target_key="jp_tokens")
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

from timeit import default_timer as timer


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == special_tokens.EOS:
            break
    return ys



if __name__ == '__main__':
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        if (epoch+1) % SAVE_EVERY == 0:
            torch.save(transformer.state_dict(), f"output/ckp_{epoch+1}.pth")

