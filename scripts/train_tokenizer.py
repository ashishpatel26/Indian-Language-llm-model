"""SentencePiece Unigram tokenizer.
usage: python scripts/train_tokenizer.py clean/hi.txt[,clean/ta.txt] 32000
"""
import os, sys
import sentencepiece as spm

files, vocab = sys.argv[1], int(sys.argv[2])
LANG_TAGS = ["<hi>", "<mr>", "<ne>", "<bn>", "<ta>", "<te>", "<gu>", "<kn>", "<ml>", "<pa>", "<or>"]
os.makedirs("tokenizer", exist_ok=True)

spm.SentencePieceTrainer.train(
    input=files,
    model_prefix="tokenizer/spm",
    model_type="unigram",
    vocab_size=vocab,
    character_coverage=0.9995,
    byte_fallback=True,
    input_sentence_size=5_000_000,
    shuffle_input_sentence=True,
    normalization_rule_name="nfkc",
    split_digits=True,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    user_defined_symbols=LANG_TAGS,
    num_threads=os.cpu_count(),
)

sp = spm.SentencePieceProcessor(model_file="tokenizer/spm.model")
s = "भारत एक विशाल देश है जिसकी संस्कृति बहुत पुरानी है।"
pieces = sp.encode(s, out_type=str)
print(f"{len(pieces)} tokens / {len(s.split())} words -> fertility {len(pieces)/len(s.split()):.2f}")
print(pieces)
assert sp.decode(sp.encode(s)) == s
