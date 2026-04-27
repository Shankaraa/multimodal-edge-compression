# Track B Calibration Text Set

## Conclusion

Step 2 is prepared, but calibration has not been run.

The calibration artifact is a text-only multilingual JSONL file built from FLEURS transcripts:

- `data/calibration/track_b_multilingual_text_256.jsonl`
- `data/calibration/track_b_multilingual_text_256.summary.json`

## Source

- Source: `google/fleurs`
- Split: `test`
- Access path: repo streaming helper in `src/voxtral_project/dataset_utils.py`
- Builder: `scripts/build_track_b_calibration_text.py`

Build command:

```bash
python scripts/build_track_b_calibration_text.py \
  --dataset-source google_fleurs \
  --total-samples 256
```

## Coverage

The set contains 256 records across all 13 requested languages:

| Language | FLEURS config | Records |
|---|---:|---:|
| EN | `en_us` | 20 |
| ZH | `cmn_hans_cn` | 20 |
| HI | `hi_in` | 20 |
| ES | `es_419` | 20 |
| AR | `ar_eg` | 20 |
| FR | `fr_fr` | 20 |
| PT | `pt_br` | 20 |
| RU | `ru_ru` | 20 |
| DE | `de_de` | 20 |
| JA | `ja_jp` | 19 |
| KO | `ko_kr` | 19 |
| IT | `it_it` | 19 |
| NL | `nl_nl` | 19 |

Passage mix:

- Short records: 126
- Medium records: 130
- Medium records are composed from 3, 4, or 5 consecutive FLEURS transcript sentences.

Validation:

- Empty texts: 0
- Duplicate texts: 0
- Short `source_ids` length: 1 for all short records
- Medium `source_ids` lengths: 52 records with 3, 39 with 4, 39 with 5
- Character range: 22 to 952
- Average characters per record: 304.2

## Tokenizer Check

Tokenizer validation file:

- `data/calibration/track_b_multilingual_text_256.tokenizer_check.json`

Validation command used the exact local Voxtral tokenizer:

```python
AutoTokenizer.from_pretrained("models/voxtral-realtime")
```

Resolved tokenizer:

- Class: `MistralCommonBackend`
- Module: `transformers.tokenization_mistral_common`
- Source: `models/voxtral-realtime`
- Vocab size: 131072

Tokenizer validation result:

- Records loaded: 256
- Minimum records per language: 19
- Tokenization errors: 0
- Minimum tokens: 12
- Maximum tokens: 403
- Average tokens: 91.55

## Schema

Each JSONL row contains:

- `id`
- `language`
- `dataset_config`
- `source`
- `split`
- `kind`
- `source_ids`
- `text`

The field to pass to the tokenizer/collator is `text`.

## Calibration Note

Text calibration is acceptable for this Track B W4A16 decoder GPTQ recipe because only the decoder
projection weights are targeted. It is still an approximation: inference decoder activations are
conditioned by projected audio embeddings and realtime delay conditioning, so the calibration data
should be treated as a pragmatic GPTQ set rather than a perfect speech-path activation match.
