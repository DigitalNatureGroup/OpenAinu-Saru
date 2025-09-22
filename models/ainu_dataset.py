# EOS:2, BOS: 1,PAD: -1, UNK: 0

from ainuseq23.to_seq import AINU_VOCAB_HEAD_SIZE, MASK_ID
#from collections import
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from ainuseq23.bert_module import device
from ainuseq23.tokenizer.ainu_tokenizer import special_tokens
import numpy as np
import typing
# EOS:2, BOS: 1,PAD: -1, UNK: 0
special_tokens_set = set([2, 1, -1, 0])

mask_rate = 0.70
unmask_alt = 0.5
unmask_keep = 0.5
ainu_vocab_size = 1250 # Ainu MAX
en_vocab_size = 32000 + ainu_vocab_size# EN MAX

class AinuFolk_JSON_MLM_Dataset(Dataset):
    def __init__(self,json_file_path,randmask = False, reverse = False ):
        self.randmask = randmask
        self.reverse = reverse
        with open(json_file_path,"r") as fp:
            self.dataset_obj = json.load(fp)
        self.dataset_len = len(self.dataset_obj)
        self.ainu_vocab_set = set()
        self.en_vocab_set = set()
        for sentence in self.dataset_obj:
            self.ainu_vocab_set = self.ainu_vocab_set | set(sentence["ainu_tokens"])
            self.en_vocab_set = self.en_vocab_set | set(sentence["en_tokens"])
        self.ainu_vocab_set = self.ainu_vocab_set - special_tokens_set
        self.en_vocab_set = self.en_vocab_set - special_tokens_set

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        sentence = self.dataset_obj[idx]
        en_tokens = sentence["en_tokens"]
        ainu_tokens = sentence["ainu_tokens"]

        if self.randmask:

            en_mask = self.mask_tokens_random(en_tokens, self.en_vocab_set)

            en_tensor = torch.Tensor(en_tokens).long()
            en_mask_tensor = torch.Tensor(en_mask).bool()

            ainu_mask = self.mask_tokens_random(en_tokens, self.en_vocab_set)

            ainu_tensor = torch.Tensor(ainu_tokens).long()
            ainu_mask_tensor = torch.Tensor(ainu_mask).bool()


            #attention_mask = (en_tensor == self.vocab[self.PAD]).unsqueeze(0)


            #nsp_target = torch.Tensor(t)

            """
            return (
                en_tensor.to(device),
                attention_mask.to(device),
                token_mask.to(device),
                mask_target.to(device),
                nsp_target.to(device)
            )
            """
            if self.reverse:
                return (
                    ainu_tensor,
                    ainu_mask_tensor,
                    en_tensor,
                    en_mask_tensor
                )
            return (
                en_tensor,
                en_mask_tensor,
                ainu_tensor,
                ainu_mask_tensor
            )
        else:
            en_tensor = torch.Tensor(en_tokens).long()
            ainu_tensor = torch.Tensor(ainu_tokens).long()
            if self.reverse:
                return (
                    ainu_tensor,
                    en_tensor
                )
            return (
                en_tensor,
                ainu_tensor
            )

    def mask_tokens_random(self, tokens_list, token_set):
        idx = random.randint(1, len(tokens_list) - 1)

        mask = [0 for i in range(len(tokens_list))]

        if random.random() < mask_rate:
            # mask with [mask]
            mask[idx] = 1
            tokens_list[idx] = special_tokens.MASK
        elif random.random() < unmask_alt:

            mask[idx] = 1
            tokens_list[idx] = random.choice(list(token_set))
        else:
            #Do  nothing
            pass
        return mask

    def get_vocab_len(self):
        if self.reverse:
            return ainu_vocab_size, en_vocab_size
        return en_vocab_size, ainu_vocab_size

    def _update_length(self, sentences: typing.List[str], lengths: typing.List[int]):
        for v in sentences:
            l = len(v.split())
            lengths.append(l)
        return lengths

    def _find_optimal_sentence_length(self, lengths: typing.List[int]):
        arr = np.array(lengths)
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

    """
    def _fill_vocab(self):
        # specials= argument is only in 0.12.0 version
        # specials=[self.CLS, self.PAD, self.MASK, self.SEP, self.UNK]
        self.vocab = vocab(self.counter, min_freq=2)

        # 0.11.0 uses this approach to insert specials
        self.vocab.insert_token(self.CLS, 0)
        self.vocab.insert_token(self.PAD, 1)
        self.vocab.insert_token(self.MASK, 2)
        self.vocab.insert_token(self.SEP, 3)
        self.vocab.insert_token(self.UNK, 4)
        self.vocab.set_default_index(4)
    """


    def _create_item(self, first: typing.List[str], second: typing.List[str], target: int = 1):
        # Create masked sentence item
        updated_first, first_mask = self._preprocess_sentence(first.copy())
        updated_second, second_mask = self._preprocess_sentence(second.copy())

        nsp_sentence = updated_first + [self.SEP] + updated_second
        nsp_indices = self.vocab.lookup_indices(nsp_sentence)
        inverse_token_mask = first_mask + [True] + second_mask

        # Create sentence item without masking random words
        first, _ = self._preprocess_sentence(first.copy(), should_mask=False)
        second, _ = self._preprocess_sentence(second.copy(), should_mask=False)
        original_nsp_sentence = first + [self.SEP] + second
        original_nsp_indices = self.vocab.lookup_indices(original_nsp_sentence)

        if self.should_include_text:
            return (
                nsp_sentence,
                nsp_indices,
                original_nsp_sentence,
                original_nsp_indices,
                inverse_token_mask,
                target
            )
        else:
            return (
                nsp_indices,
                original_nsp_indices,
                inverse_token_mask,
                target
            )

    def _select_false_nsp_sentences(self, sentences: typing.List[str]):
        """Select sentences to create false NSP item

        Args:
            sentences: list of all sentences

        Returns:
            tuple of two sentences. The second one NOT the next sentence
        """
        sentences_len = len(sentences)
        sentence_index = random.randint(0, sentences_len - 1)
        next_sentence_index = random.randint(0, sentences_len - 1)

        # To be sure that it's not real next sentence
        while next_sentence_index == sentence_index + 1:
            next_sentence_index = random.randint(0, sentences_len - 1)

        return sentences[sentence_index], sentences[next_sentence_index]

    def _preprocess_sentence(self, sentence_tokens: typing.List[int], should_mask: bool = True):
        inverse_token_mask = None
        if should_mask:
            sentence_tokens, inverse_token_mask = self._mask_sentence(sentence_tokens)
        sentence_tokens, inverse_token_mask = self._pad_sentence([self.CLS] + sentence_tokens, [True] + inverse_token_mask)

        return sentence_tokens, inverse_token_mask

    def _mask_sentence(self, sentence: typing.List[str]):
        """Replace MASK_PERCENTAGE (15%) of words with special [MASK] symbol
        or with random word from vocabulary

        Args:
            sentence: sentence to process

        Returns:
            tuple of processed sentence and inverse token mask
        """
        len_s = len(sentence)
        inverse_token_mask = [True for _ in range(max(len_s, self.optimal_sentence_length))]

        mask_amount = round(len_s * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)

            if random.random() < 0.8:
                sentence[i] = self.MASK
            else:
                # All is below 5 is special token
                # see self._insert_specials method
                j = random.randint(5, len(self.vocab) - 1)
                sentence[i] = self.vocab.lookup_token(j)
            inverse_token_mask[i] = False
        return sentence, inverse_token_mask

    def _pad_sentence(self, sentence: typing.List[str], inverse_token_mask: typing.List[bool] = None):
        len_s = len(sentence)

        if len_s >= self.optimal_sentence_length:
            s = sentence[:self.optimal_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - len_s)

        # inverse token mask should be padded as well
        if inverse_token_mask:
            len_m = len(inverse_token_mask)
            if len_m >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[:self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (self.optimal_sentence_length - len_m)
        return s, inverse_token_mask


class Ainu_JSON_MLM_Dataset(Dataset):
    def __init__(self,json_file_path,
                 randmask:bool = False,
                 source_key:str = "ainu_tokens",
                 target_key:str = "en_tokens"):
        self.randmask = randmask
        self.source_key = source_key
        self.target_key = target_key
        with open(json_file_path,"r") as fp:
            self.dataset_obj = json.load(fp)
        self.dataset_len = len(self.dataset_obj)
        self.source_vocab_set = set()
        self.target_vocab_set = set()
        for sentence in self.dataset_obj:
            self.source_vocab_set = self.source_vocab_set | set(sentence[source_key])
            self.target_vocab_set = self.target_vocab_set | set(sentence[target_key])
        self.source_vocab_set = self.source_vocab_set - special_tokens_set
        self.target_vocab_set = self.target_vocab_set - special_tokens_set

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        sentence = self.dataset_obj[idx]
        target_tokens = sentence[self.target_key]
        source_tokens = sentence[self.source_key]

        if self.randmask:

            target_mask = self.mask_tokens_random(target_tokens, self.target_vocab_set)

            target_tensor = torch.Tensor(target_tokens).long()
            target_mask_tensor = torch.Tensor(target_mask).bool()

            source_mask = self.mask_tokens_random(target_tokens, self.target_vocab_set)

            source_tensor = torch.Tensor(source_tokens).long()
            source_mask_tensor = torch.Tensor(source_mask).bool()


            #attention_mask = (en_tensor == self.vocab[self.PAD]).unsqueeze(0)


            #nsp_target = torch.Tensor(t)

            """
            return (
                en_tensor.to(device),
                attention_mask.to(device),
                token_mask.to(device),
                mask_target.to(device),
                nsp_target.to(device)
            )
            """
            return (
                source_tensor,
                source_mask_tensor,
                target_tensor,
                target_mask_tensor
            )

        else:
            target_tensor = torch.Tensor(target_tokens).long()
            source_tensor = torch.Tensor(source_tokens).long()

            return (
                source_tensor,
                target_tensor
            )


    def mask_tokens_random(self, tokens_list, token_set):
        idx = random.randint(1, len(tokens_list) - 1)

        mask = [0 for i in range(len(tokens_list))]

        if random.random() < mask_rate:
            # mask with [mask]
            mask[idx] = 1
            tokens_list[idx] = special_tokens.MASK
        elif random.random() < unmask_alt:

            mask[idx] = 1
            tokens_list[idx] = random.choice(list(token_set))
        else:
            #Do  nothing
            pass
        return mask

    def get_vocab_len(self):
        return len(self.source_vocab_set), len(self.target_vocab_set)

    def _update_length(self, sentences: typing.List[str], lengths: typing.List[int]):
        for v in sentences:
            l = len(v.split())
            lengths.append(l)
        return lengths

    def _find_optimal_sentence_length(self, lengths: typing.List[int]):
        arr = np.array(lengths)
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

    """
    def _fill_vocab(self):
        # specials= argument is only in 0.12.0 version
        # specials=[self.CLS, self.PAD, self.MASK, self.SEP, self.UNK]
        self.vocab = vocab(self.counter, min_freq=2)

        # 0.11.0 uses this approach to insert specials
        self.vocab.insert_token(self.CLS, 0)
        self.vocab.insert_token(self.PAD, 1)
        self.vocab.insert_token(self.MASK, 2)
        self.vocab.insert_token(self.SEP, 3)
        self.vocab.insert_token(self.UNK, 4)
        self.vocab.set_default_index(4)
    """


    def _create_item(self, first: typing.List[str], second: typing.List[str], target: int = 1):
        # Create masked sentence item
        updated_first, first_mask = self._preprocess_sentence(first.copy())
        updated_second, second_mask = self._preprocess_sentence(second.copy())

        nsp_sentence = updated_first + [self.SEP] + updated_second
        nsp_indices = self.vocab.lookup_indices(nsp_sentence)
        inverse_token_mask = first_mask + [True] + second_mask

        # Create sentence item without masking random words
        first, _ = self._preprocess_sentence(first.copy(), should_mask=False)
        second, _ = self._preprocess_sentence(second.copy(), should_mask=False)
        original_nsp_sentence = first + [self.SEP] + second
        original_nsp_indices = self.vocab.lookup_indices(original_nsp_sentence)

        if self.should_include_text:
            return (
                nsp_sentence,
                nsp_indices,
                original_nsp_sentence,
                original_nsp_indices,
                inverse_token_mask,
                target
            )
        else:
            return (
                nsp_indices,
                original_nsp_indices,
                inverse_token_mask,
                target
            )

    def _select_false_nsp_sentences(self, sentences: typing.List[str]):
        """Select sentences to create false NSP item

        Args:
            sentences: list of all sentences

        Returns:
            tuple of two sentences. The second one NOT the next sentence
        """
        sentences_len = len(sentences)
        sentence_index = random.randint(0, sentences_len - 1)
        next_sentence_index = random.randint(0, sentences_len - 1)

        # To be sure that it's not real next sentence
        while next_sentence_index == sentence_index + 1:
            next_sentence_index = random.randint(0, sentences_len - 1)

        return sentences[sentence_index], sentences[next_sentence_index]

    def _preprocess_sentence(self, sentence_tokens: typing.List[int], should_mask: bool = True):
        inverse_token_mask = None
        if should_mask:
            sentence_tokens, inverse_token_mask = self._mask_sentence(sentence_tokens)
        sentence_tokens, inverse_token_mask = self._pad_sentence([self.CLS] + sentence_tokens, [True] + inverse_token_mask)

        return sentence_tokens, inverse_token_mask

    def _mask_sentence(self, sentence: typing.List[str]):
        """Replace MASK_PERCENTAGE (15%) of words with special [MASK] symbol
        or with random word from vocabulary

        Args:
            sentence: sentence to process

        Returns:
            tuple of processed sentence and inverse token mask
        """
        len_s = len(sentence)
        inverse_token_mask = [True for _ in range(max(len_s, self.optimal_sentence_length))]

        mask_amount = round(len_s * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)

            if random.random() < 0.8:
                sentence[i] = self.MASK
            else:
                # All is below 5 is special token
                # see self._insert_specials method
                j = random.randint(5, len(self.vocab) - 1)
                sentence[i] = self.vocab.lookup_token(j)
            inverse_token_mask[i] = False
        return sentence, inverse_token_mask

    def _pad_sentence(self, sentence: typing.List[str], inverse_token_mask: typing.List[bool] = None):
        len_s = len(sentence)

        if len_s >= self.optimal_sentence_length:
            s = sentence[:self.optimal_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - len_s)

        # inverse token mask should be padded as well
        if inverse_token_mask:
            len_m = len(inverse_token_mask)
            if len_m >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[:self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (self.optimal_sentence_length - len_m)
        return s, inverse_token_mask