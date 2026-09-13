# BFCL r1 — invalid adapter startup

All 20 tasks failed before issuing a model request: Transformers 5 returned a
BatchEncoding instead of an integer token list. Not a model accuracy or
throughput result. Fixed with explicit `return_dict=False` and a real-tokenizer
serialization test. Model process group and per-run containers cleaned up.
