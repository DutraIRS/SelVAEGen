import re

import selfies as sf
import torch


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"
ENC_TOKEN = "<enc>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN, ENC_TOKEN]

SMILES_TOKEN_RE = re.compile(r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@@|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")


class BaseTokenizer:
    def __init__(self, vocab):
        tokens = list(SPECIAL_TOKENS)
        tokens += [token for token in vocab if token not in SPECIAL_TOKENS]

        self.tokens = tokens
        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.mask_id = self.token_to_id[MASK_TOKEN]
        self.enc_id = self.token_to_id[ENC_TOKEN]
        self._content_ids = None

    def __len__(self):
        return len(self.tokens)

    @property
    def vocab_size(self):
        return len(self.tokens)

    @property
    def content_ids(self):
        """Ids that decode to a real molecule on their own."""
        if self._content_ids is None:
            from rdkit import Chem

            self._content_ids = tuple(
                index for index, token in enumerate(self.tokens)
                if token not in SPECIAL_TOKENS
                and (text := self._detokenize([token]))
                and Chem.MolFromSmiles(text) is not None)

        return self._content_ids

    def _tokenize(self, value):
        raise NotImplementedError

    def _detokenize(self, tokens):
        raise NotImplementedError

    def encode(self, value, max_length=None, add_special_tokens=True, device=None):
        tokens = self._tokenize(value)
        if tokens is None:
            return None

        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if self.unk_id in ids:
            return None

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        if max_length is not None:
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            if len(ids) > max_length:
                return None
            ids += [self.pad_id] * (max_length - len(ids))

        return torch.tensor(ids, dtype=torch.long, device=device)

    def decode(self, ids, strip_bos=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()

        tokens = []
        for position, token_id in enumerate(ids):
            token = self.id_to_token.get(int(token_id), UNK_TOKEN)

            if token == EOS_TOKEN:
                break
            if token == PAD_TOKEN:
                continue
            if strip_bos and position == 0 and token == BOS_TOKEN:
                continue
            if token == UNK_TOKEN:
                return ""

            tokens.append(token)

        return self._detokenize(tokens)

    def decode_batch(self, batch, strip_bos=True):
        return [self.decode(row, strip_bos=strip_bos) for row in batch]

class SmilesTokenizer(BaseTokenizer):
    @staticmethod
    def tokenize(smiles):
        if not isinstance(smiles, str) or not smiles:
            return None

        tokens = SMILES_TOKEN_RE.findall(smiles)

        # Regex findall must consume the entire SMILES
        if "".join(tokens) != smiles:
            return None

        return tokens

    @classmethod
    def fit(cls, smiles_iterable):
        vocab = set()

        for smiles in smiles_iterable:
            tokens = cls.tokenize(smiles)
            if tokens is not None:
                vocab.update(tokens)

        return cls(sorted(vocab))

    def _tokenize(self, smiles):
        return self.tokenize(smiles)

    def _detokenize(self, tokens):
        return "".join(tokens)


class SelfiesTokenizer(BaseTokenizer):
    def __init__(self, extra=()):
        self.extra = sorted(extra)
        super().__init__(sorted(set(sf.get_semantic_robust_alphabet()) | set(extra)))

    @staticmethod
    def tokenize(smiles):
        if not isinstance(smiles, str) or not smiles:
            return None

        try:
            selfies = sf.encoder(smiles)
            return list(sf.split_selfies(selfies))
        except Exception:
            return None

    @staticmethod
    def keeps_decoder_total(alphabet, trials=200, length=30, seed=0):
        """Did `trials` random sequences over this alphabet all decode to a real molecule?"""
        import numpy as np
        from rdkit import Chem

        alphabet = sorted(alphabet)
        rng = np.random.default_rng(seed)

        for _ in range(trials):
            try:
                smiles = sf.decoder("".join(rng.choice(alphabet, length)))
            except Exception:
                return False

            if not smiles or Chem.MolFromSmiles(smiles) is None:
                return False

        return True

    @classmethod
    def fit(cls, smiles_iterable, trials=200, verify_trials=20000, rounds=3, seed=0):
        """The robust alphabet, widened by data tokens a random-sequence screen keeps total."""
        robust = set(sf.get_semantic_robust_alphabet())

        encoded = []
        for smiles in smiles_iterable:
            tokens = cls.tokenize(smiles)
            if tokens is not None:
                encoded.extend(tokens)

        extra = [token for token in sorted(set(encoded) - robust - {"."})
                    if cls.keeps_decoder_total(robust | {token}, trials=trials)]

        clean = 0
        while extra and clean < rounds:
            suspects = cls._suspects(robust, extra, trials=verify_trials, seed=seed + clean + 1)
            if not suspects:
                clean += 1
                continue

            guilty = [token for token in suspects if not cls.keeps_decoder_total(robust | {token}, trials=verify_trials, seed=seed + clean + 1)]
            extra = [token for token in extra if token not in set(guilty or suspects)]
            clean = 0

        return cls(extra=extra)

    @classmethod
    def _suspects(cls, robust, extra, trials, seed):
        """Added tokens appearing in any failing sequence, sorted so the result is stable."""
        import numpy as np
        from rdkit import Chem

        alphabet = sorted(robust | set(extra))
        added = set(extra)
        rng = np.random.default_rng(seed)
        blamed = set()

        for _ in range(trials):
            tokens = list(rng.choice(alphabet, 30))
            try:
                smiles = sf.decoder("".join(tokens))
                broken = not smiles or Chem.MolFromSmiles(smiles) is None
            except Exception:
                broken = True

            if broken:
                blamed |= set(tokens) & added

        return sorted(blamed)

    def _tokenize(self, smiles):
        return self.tokenize(smiles)

    def _detokenize(self, tokens):
        selfies = "".join(tokens)

        try:
            return sf.decoder(selfies)
        except Exception:
            return ""
