"""
Task 1: do the three pretrained embedding models fit IMDb?

The three models, all 300-dimensional, stored in embeddings/ (not tracked by git):

    glove     GloVe 2024 WikiGiga 300d       glove.2024.wikigiga.300d.zip   (text)
    fasttext  fastText Common Crawl 2M 300d  crawl-300d-2M.vec.zip          (text, first line is a header)
    word2vec  word2vec Google News 300d      word2vec-google-news-300.gz    (binary)

IMDb reviews are split into words exactly as in part 1 (part1/tokenisation.py: the same cleanup and
word splitting), but without part 1's vocabulary: each embedding model's own vocabulary decides
which words are known. Words are looked up the way part2.clslib.PretrainedEmbeddings does it:
lowercased, and when two entries lowercase to the same word, the first one in the file wins.

Every model gets the same split text. Before the lookup, `adapt_tokens` rewrites a few tokens into
the convention each model was trained with (see MODEL_RULES). A mismatch is fixed only if it is a
spelling convention (not a missing word), affects negation or a measurable share of tokens, and the
fix keeps the meaning of the text.

Usage, from the repository root:

    python part2/task1_choose_embeddings/embedding_check.py conventions   # how each model writes contractions, numbers, ...
    python part2/task1_choose_embeddings/embedding_check.py coverage      # coverage of IMDb, before and after adapt_tokens
    python part2/task1_choose_embeddings/embedding_check.py export        # write embeddings/imdb_<model>.txt.gz (IMDb words only)

The first run of either command reads each model file once and caches its word list in
embeddings/vocab_<model>.txt (one word per line, in file order).
"""
import argparse
import gzip
import re
import sys
from collections import Counter
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]  # the repository root (Athena/)
EMB_DIR = ROOT / "embeddings"
DIMS = 300

MODELS = {
    "glove": {"file": "glove.2024.wikigiga.300d.zip", "format": "text", "header": False},
    "fasttext": {"file": "crawl-300d-2M.vec.zip", "format": "text", "header": True},
    "word2vec": {"file": "word2vec-google-news-300.gz", "format": "binary", "header": True},
}


# ---------------------------------------------------------------- reading the model files

def _open_text(path):
    """Text stream of a .zip (with exactly one member), .gz or plain text file."""
    if path.suffix == ".zip":
        archive = ZipFile(path)
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(f"Expected one file in {path}, found {members}")
        return TextIOWrapper(archive.open(members[0]), encoding="utf-8", errors="replace")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_text_entries(path, header):
    """
    Yields (word, values) for each line "word v1 ... v300" of a text embedding file, where values
    is the unparsed string of numbers. Lines that PretrainedEmbeddings would skip (a word that
    contains whitespace, or the wrong number of values) are skipped here too.
    """
    with _open_text(path) as f:
        if header:
            next(f)
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", DIMS)
            if len(parts) != DIMS + 1 or not parts[0] or len(parts[0].split()) != 1:
                continue
            yield parts[0], " ".join(parts[1:])


def iter_binary_entries(path, chunk_size=1 << 22):
    """
    Yields (word, raw_bytes) for each entry of a word2vec binary file: a header line
    "<num_words> <dims>", then for each entry the word, a space and dims float32 values.
    """
    with gzip.open(path, "rb") as f:
        num_words, dims = map(int, f.readline().split())
        vec_bytes = 4 * dims
        buf, pos = f.read(chunk_size), 0
        for _ in range(num_words):
            space = buf.find(b" ", pos)
            while space < 0 or space + 1 + vec_bytes > len(buf):
                more = f.read(chunk_size)
                if not more:
                    raise ValueError(f"{path} ended early")
                buf, pos = buf[pos:] + more, 0
                space = buf.find(b" ", pos)
            word = buf[pos:space].lstrip(b"\n").decode("utf-8", errors="replace")
            yield word, buf[space + 1:space + 1 + vec_bytes]
            pos = space + 1 + vec_bytes


def fetch_vectors(name, words):
    """Vectors (numpy float32) of the given lowercase words, read from the full model file."""
    import numpy as np
    wanted, found = set(words), {}
    for word, values in iter_entries(name):
        word = word.lower()
        if word in wanted and word not in found:
            if isinstance(values, bytes):
                found[word] = np.frombuffer(values, dtype=np.float32)
            else:
                found[word] = np.array(values.split(), dtype=np.float32)
    return found


def iter_entries(name):
    spec = MODELS[name]
    path = EMB_DIR / spec["file"]
    if spec["format"] == "binary":
        return iter_binary_entries(path)
    return iter_text_entries(path, spec["header"])


def load_words(name):
    """The model's words in file order (original case), cached in embeddings/vocab_<name>.txt."""
    cache = EMB_DIR / f"vocab_{name}.txt"
    if not cache.exists():
        print(f"Reading the vocabulary of {name} (first time only)...", file=sys.stderr)
        words = [word for word, _ in iter_entries(name)]
        cache.write_text("\n".join(words), encoding="utf-8")
    return cache.read_text(encoding="utf-8").split("\n")


def lowercase_lookup(words):
    """Lowercased word -> rank (1-based) of the first entry with that lowercased form."""
    lookup = {}
    for rank, word in enumerate(words, 1):
        lookup.setdefault(word.lower(), rank)
    return lookup


# ---------------------------------------------------------------- IMDb, split as in part 1

def imdb_splitter():
    """Function text -> list of words: part 1's cleanup and word splitting, no vocabulary."""
    sys.path.insert(0, str(ROOT))
    from part1.tokenisation import get_tokenizer
    tokenizer = get_tokenizer("word", 32000, folder=str(ROOT / "results" / "tokenizers"))
    normalizer, pre_tokenizer = tokenizer.normalizer, tokenizer.pre_tokenizer

    def split(text):
        return [word for word, _ in pre_tokenizer.pre_tokenize_str(normalizer.normalize_str(text))]
    return split


class ImdbTokenizer:
    """
    Tokenizer for tasks 3-4, usable wherever the tutorial expects one (part2.experiment.train,
    predict_logprobs): part 1's cleanup and word splitting, then adapt_tokens for `model`, then the
    ids of `emb_model` (a PretrainedEmbeddings; unknown words get its unk id).

        emb = PretrainedEmbeddings("embeddings/imdb_fasttext.txt.gz", 300)
        tokenizer = ImdbTokenizer(emb, "fasttext")
    """

    def __init__(self, emb_model, model):
        self.vocab = emb_model.vocab
        self.model = model
        self._split = imdb_splitter()

    def tokens(self, text):
        return adapt_tokens(self._split(text), self.model)

    def ids(self, text):
        return [self.vocab.index(t) for t in self.tokens(text)]


def token_kind(token):
    if re.search(r"[^\W\d_]", token):
        return "word"      # contains a letter
    if re.search(r"\d", token):
        return "number"
    return "punctuation"


# ---------------------------------------------------------------- per-model conventions

# What each rule fixes, measured on the IMDb train split (embedding_check.py conventions / coverage):
#   glove     its tokenizer split "cannot" into "can" "not"; ours keeps it whole (1,097 tokens)
#   fasttext  it writes didn't as "didn" "'t" and has no "n't" (33,559 tokens, i.e. most negations),
#             so did + n't becomes didn + 't; its "ca" and "wo" are other words (circa, ...), so
#             before "n't" they become can / won (can't -> can 't, won't -> won 't)
#   word2vec  its "n't" is a rare, noisy entry, so we use "not"; ca / wo before "n't" become
#             can / will; numbers with 2+ digits were stored with # for each digit (10 -> ##);
#             it has no punctuation at all, so punctuation is dropped instead of becoming unknown
MODEL_RULES = {
    "glove": {"replace": {"cannot": ["can", "not"]}},
    "fasttext": {"replace": {"n't": ["'t"]}, "before_nt": {"ca": "can", "wo": "won"}, "add_n_before_nt": True},
    "word2vec": {"replace": {"n't": ["not"]}, "before_nt": {"ca": "can", "wo": "will", "sha": "shall"},
                 "hash_digits": True, "drop_punctuation": True},
}


def adapt_tokens(tokens, model):
    """Rewrite tokens split as in part 1 into the convention of `model` (see MODEL_RULES)."""
    rules = MODEL_RULES[model]
    replace = rules.get("replace", {})
    before_nt = rules.get("before_nt", {})
    adapted = []
    for i, token in enumerate(tokens):
        if i + 1 < len(tokens) and tokens[i + 1] == "n't" and token in before_nt:
            adapted.append(before_nt[token])
        elif i + 1 < len(tokens) and tokens[i + 1] == "n't" and rules.get("add_n_before_nt"):
            adapted.append(token + "n")
        elif token in replace:
            adapted.extend(replace[token])
        elif rules.get("drop_punctuation") and token_kind(token) == "punctuation":
            continue
        elif rules.get("hash_digits") and sum(c.isdigit() for c in token) >= 2:
            adapted.append(re.sub(r"\d", "#", token))
        else:
            adapted.append(token)
    return adapted


# ---------------------------------------------------------------- the two reports

PROBES = {
    "contractions": ["n't", "did", "didn't", "didn", "'t", "t", "ca", "can't", "wo", "won't",
                     "'s", "it's", "'re", "'ll", "'ve", "'m", "'d"],
    "punctuation": [".", ",", "!", "?", "...", "…", "!!", "?!", '"', "``", "''", "'", "-", "--",
                    "(", "-lrb-", ":", ";", "/", "&", "*", ":)"],
    "hyphens": ["well-made", "over-the-top", "b-movie", "must-see", "low-budget", "made-for-tv",
                "well_made", "over_the_top", "must_see", "low_budget"],
    "numbers": ["8", "10", "8/10", "7/10", "1/10", "2", "1990", "1990s", "80s", "#", "##", "####",
                "#/##", "##s"],
    "function words": ["a", "the", "of", "to", "and", "is", "not", "no", "never", "but"],
    "case": ["good", "Good", "GOOD", "kubrick", "Kubrick", "imdb", "IMDb", "IMDB"],
    "phrases": ["New_York", "special_effects", "waste_of_time", "must_see", "worth_watching"],
    "review words": ["movie", "film", "cheesy", "meh", "lame", "rewatchable", "unwatchable",
                     "overacted", "underrated", "overrated", "cringe", "spoilers"],
}


def conventions(names):
    vocab = {name: load_words(name) for name in names}
    exact = {name: {w: r for r, w in reversed(list(enumerate(words, 1)))} for name, words in vocab.items()}
    lower = {name: lowercase_lookup(words) for name, words in vocab.items()}

    print("Vocabulary size and what kind of entries each model has")
    rows = {
        "entries": lambda ws: len(ws),
        "with an uppercase letter": lambda ws: sum(any(c.isupper() for c in w) for w in ws),
        "lowercase forms (what the loader keeps)": lambda ws: len({w.lower() for w in ws}),
        "with a digit": lambda ws: sum(any(c.isdigit() for c in w) for w in ws),
        "with #": lambda ws: sum("#" in w for w in ws),
        "with _ (phrases)": lambda ws: sum("_" in w for w in ws),
        "with a hyphen between letters": lambda ws: sum(bool(re.search(r"[^\W\d_]-[^\W\d_]", w)) for w in ws),
        "punctuation only": lambda ws: sum(token_kind(w) == "punctuation" for w in ws),
    }
    print(f"{'':42}" + "".join(f"{n:>12}" for n in names))
    for label, fn in rows.items():
        print(f"{label:42}" + "".join(f"{fn(vocab[n]):>12,}" for n in names))

    print("\nProbes: rank of the exact entry (1 = most frequent); (lc N) = only a different case, at rank N; - = absent")
    for group, probes in PROBES.items():
        print(f"\n[{group}]")
        for probe in probes:
            cells = []
            for n in names:
                if probe in exact[n]:
                    cells.append(f"{exact[n][probe]:,}")
                elif probe.lower() in lower[n]:
                    cells.append(f"(lc {lower[n][probe.lower()]:,})")
                else:
                    cells.append("-")
            print(f"  {probe:18}" + "".join(f"{c:>16}" for c in cells))


def coverage_stats(reviews, known, lexicon, min_count=5):
    """
    Coverage of tokenized reviews by a model's vocabulary `known` (lowercased words).
    Returns (dict of percentages, Counter of unknown tokens).
    """
    counts = Counter(token for review in reviews for token in review)
    total = sum(counts.values())
    kinds, unknown_kinds = Counter(), Counter()
    unknown = Counter()
    for token, c in counts.items():
        kinds[token_kind(token)] += c
        if token not in known:
            unknown[token] = c
            unknown_kinds[token_kind(token)] += c
    frequent = [t for t, c in counts.items() if c >= min_count]
    lexicon_counts = {w: c for w, c in counts.items() if w in lexicon}
    lexicon_known = [w for w in lexicon_counts if w in known]
    reviews_over_5 = sum(sum(t not in known for t in r) > 0.05 * len(r) for r in reviews if r)

    def share_known(kind):
        return 100 * (1 - unknown_kinds[kind] / kinds[kind]) if kinds[kind] else float("nan")

    stats = {
        "tokens looked up (millions)": total / 1e6,
        "token coverage": 100 * (1 - sum(unknown.values()) / total),
        "  word tokens": share_known("word"),
        "  number tokens": share_known("number"),
        "  punctuation tokens": share_known("punctuation"),
        "type coverage (all distinct tokens)": 100 * (1 - len(unknown) / len(counts)),
        f"type coverage (tokens seen >= {min_count} times)": 100 * sum(t in known for t in frequent) / len(frequent),
        "reviews with > 5% unknown tokens": 100 * reviews_over_5 / len(reviews),
        "opinion-lexicon types covered": 100 * len(lexicon_known) / len(lexicon_counts),
        "opinion-lexicon tokens covered": 100 * sum(lexicon_counts[w] for w in lexicon_known) / sum(lexicon_counts.values()),
    }
    return stats, unknown


def coverage(names, split_name="train", top_unknown=40):
    """Coverage of IMDb by each model, before and after adapt_tokens."""
    from datasets import load_dataset
    from nltk.corpus import opinion_lexicon
    split = imdb_splitter()
    texts = load_dataset("stanfordnlp/imdb", split=split_name)["text"]
    reviews = [split(text) for text in texts]
    lexicon = set(opinion_lexicon.positive()) | set(opinion_lexicon.negative())
    print(f"IMDb {split_name}: {len(reviews):,} reviews, split as in part 1\n")

    columns, unknown_after = {}, {}
    for n in names:
        known = lowercase_lookup(load_words(n))
        columns[f"{n} raw"], _ = coverage_stats(reviews, known, lexicon)
        columns[f"{n} fixed"], unknown_after[n] = coverage_stats([adapt_tokens(r, n) for r in reviews], known, lexicon)

    print(f"{'(%)':48}" + "".join(f"{c:>15}" for c in columns))
    for label in columns[next(iter(columns))]:
        print(f"{label:48}" + "".join(f"{col[label]:>15.2f}" for col in columns.values()))

    for n in names:
        words = Counter({t: c for t, c in unknown_after[n].items() if token_kind(t) == "word"})
        print(f"\nMost frequent unknown tokens for {n}, after the fixes (count):")
        print("  " + ", ".join(f"{t} ({c:,})" for t, c in unknown_after[n].most_common(top_unknown)))
        print(f"Most frequent unknown words (with a letter) for {n}, after the fixes:")
        print("  " + ", ".join(f"{t} ({c:,})" for t, c in words.most_common(top_unknown)))


def export_imdb_vectors(names, splits=("train", "test")):
    """
    For each model, write embeddings/imdb_<model>.txt.gz: the vectors of every word that occurs in
    the IMDb `splits` (split as in part 1, then adapt_tokens), in the model's file order and in the
    text format part2.clslib.PretrainedEmbeddings reads:

        emb = PretrainedEmbeddings("embeddings/imdb_glove.txt.gz", 300)

    Words are lowercased, and the first entry of each lowercased form wins, as in PretrainedEmbeddings.
    The vectors come from the pretrained model, not from IMDb, so including the test split's words
    gives the test reviews no more coverage than the full model would, and uses none of their labels.
    """
    import numpy as np
    from datasets import load_dataset
    split = imdb_splitter()
    reviews = [split(text) for name in splits for text in load_dataset("stanfordnlp/imdb", split=name)["text"]]
    for n in names:
        wanted = {token for review in reviews for token in adapt_tokens(review, n)}
        out = EMB_DIR / f"imdb_{n}.txt.gz"
        kept = set()
        with gzip.open(out, "wt", encoding="utf-8") as f:
            for word, values in iter_entries(n):
                word = word.lower()
                if word not in wanted or word in kept:
                    continue
                kept.add(word)
                if MODELS[n]["format"] == "binary":
                    values = " ".join(f"{v:.6g}" for v in np.frombuffer(values, dtype=np.float32))
                f.write(f"{word} {values}\n")
        print(f"{n}: {len(kept):,} of {len(wanted):,} IMDb word types written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", choices=["conventions", "coverage", "export"])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()
    if args.report == "conventions":
        conventions(args.models)
    elif args.report == "coverage":
        coverage(args.models)
    else:
        export_imdb_vectors(args.models)
