from transformers import PreTrainedTokenizer
import torch
import os


class DNATokenizer(PreTrainedTokenizer):
    def __init__(self, vocab=None, **kwargs):
        if vocab is None:
            vocab = ["A", "T", "C", "G", "-", "[PAD]", "[MASK]", "[CLS]", "[SEP]"]
        self.vocab = vocab
        self.tokens_to_ids = {tok: i for i, tok in enumerate(vocab)}
        self.ids_to_tokens = {i: tok for i, tok in enumerate(vocab)}
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return self.tokens_to_ids

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if filename_prefix is None:
            filename_prefix = ""
    
        vocab_file = os.path.join(
            save_directory,
            f"{filename_prefix}vocab.txt"
        )
    
        with open(vocab_file, "w") as f:
            for token in self.vocab:
                f.write(token + "\n")
    
        return (vocab_file,)


    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.tokens_to_ids.get(token)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index)

    def __call__(self, sequences, max_length=512, padding="max_length", return_tensors="pt"):
        if isinstance(sequences, str):
            sequences = [sequences]
    
        all_ids = []
        all_attention_mask = []
        for seq in sequences:
            ids = [self.tokens_to_ids[self.cls_token]] + \
                  [self._convert_token_to_id(tok) for tok in seq] + \
                  [self.tokens_to_ids[self.sep_token]]
            if len(ids) < max_length:
                attention_mask = [1]*len(ids) + [0]*(max_length - len(ids))
                ids += [self.tokens_to_ids[self.pad_token]] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
                attention_mask = [1]*max_length
            all_ids.append(ids)
            all_attention_mask.append(attention_mask)
    
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long)
            }
        else:
            return {
                "input_ids": all_ids,
                "attention_mask": all_attention_mask
            }


