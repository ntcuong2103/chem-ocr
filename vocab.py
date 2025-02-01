from typing import Dict, List


class Vocab:

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, dict_path: str = 'vocab_syms.txt', vocab_map: str = 'vocab_maps.txt') -> None:
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX
        self.word2idx["<unk>"] = self.UNK_IDX

        with open(dict_path, "r") as f:
            for line in f.readlines():
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        # self.vocab_map = {u: v for u, v in [line.strip().split('\t') for line in open('vocab_maps.txt').readlines()]}
        # self.vocab_map.update({v:v for v in self.word2idx.keys() if v not in self.vocab_map.keys()})

    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)
