"""
Task 2: the two tokenisers we compare on IMDb.

    A. "word": every vocabulary entry is a whole word; other words become [UNK]
    B. "bpe": frequent words stay whole, rare words are spelled with subword pieces (marked ##)

Both are trained only on the 50,000 unlabelled reviews (the `unsupervised` split of stanfordnlp/imdb).
All text cleaning happens inside the tokeniser, so raw reviews can be passed in directly.

Usage (e.g. one W&B run, with kind and vocab_size taken from the sweep config):

    tokenizer = get_tokenizer("bpe", 32000)   # loads it if saved, otherwise trains and saves it
    tokenizer.encode(review).ids              # token ids, as used by clslib.TokenIndicator
"""
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, Regex, normalizers, pre_tokenizers
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer


KINDS = ("word", "bpe")
VOCAB_SIZES = (8000, 16000, 32000, 64000)  # candidates, chosen later by cross-validation
MIN_FREQUENCY = 2  # as in the tutorial
UNK = "[UNK]"

def train_tokenizer(kind, vocab_size, corpus=None, min_frequency=MIN_FREQUENCY):
    """
    Constructs and trains a tokeniser.

    kind: "word" or "bpe"
    vocab_size: maximum number of distinct tokens, [UNK] included
    corpus: list of texts to learn from; defaults to the 50k unlabelled IMDb reviews
    min_frequency: words (word-level) or pairs (BPE) seen fewer times are not added to the vocabulary
    """
    if kind == "word":
        tokenizer = Tokenizer(WordLevel(unk_token=UNK))
        trainer = WordLevelTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[UNK])
    elif kind == "bpe":
        # the ## prefix marks pieces inside a word (##able), so they differ from whole words (able)
        tokenizer = Tokenizer(BPE(unk_token=UNK, continuing_subword_prefix="##"))
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[UNK],
                             continuing_subword_prefix="##")
    else:
        raise ValueError(f"Unknown kind={kind}, expected one of {KINDS}")

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace(Regex(r"<br\s*/?>"), " "),  # HTML line breaks
        # the replacements come before NFKC, which would turn ´ into a space plus an accent
        normalizers.Replace("\x96", "-"),  # dashes and ellipses mis-decoded from Windows-1252
        normalizers.Replace("\x97", "-"),
        normalizers.Replace(Regex(r"-{2,}"), " - "),  # -- and --- used as dashes, so word--word splits like word - word
        normalizers.Replace("\x85", "..."),
        normalizers.Replace("\x91", "'"),  # ‘ mis-decoded from Windows-1252
        normalizers.Replace("’", "'"),  # ’ and ´ used as apostrophes, so didn’t/didn´t split like didn't
        normalizers.Replace("´", "'"),
        normalizers.NFKC(),
        # cap repeated punctuation (after NFKC, which turns … into ...): !!!!! -> !!, ????? -> ??, ...... -> ...
        normalizers.Replace(Regex(r"!{3,}"), "!!"),
        normalizers.Replace(Regex(r"\?{3,}"), "??"),
        normalizers.Replace(Regex(r"\.{4,}"), "..."),
        normalizers.Lowercase(),
    ])
    # words, with clitics split off (didn't -> did n't, kubrick's -> kubrick 's), punctuation as its own token
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"\w+(?=n't)|n't|'(?:s|re|ve|ll|d|m)\b|\w+|[^\w\s]"), behavior="removed", invert=True)

    if corpus is None:
        corpus = load_dataset("stanfordnlp/imdb", split="unsupervised")["text"]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def get_tokenizer(kind, vocab_size, folder="results/tokenizers", min_frequency=MIN_FREQUENCY):
    """
    Returns the tokeniser saved as {folder}/{kind}_vs_{vocab_size}_mf_{min_frequency}.json,
    training and saving it first if it does not exist yet.
    """
    path = Path(folder) / f"{kind}_vs_{vocab_size}_mf_{min_frequency}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    tokenizer = train_tokenizer(kind, vocab_size, min_frequency=min_frequency)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer
