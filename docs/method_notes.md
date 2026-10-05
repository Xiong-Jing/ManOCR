# Method Notes

## Detection

The detector is an official-style DBNet++ baseline:

- ResNet backbone
- DBFPN neck
- ASF scale fusion
- DBHead with probability, threshold, and binary maps

Project-specific additions:

- VSAA: vertical spatial attention aggregation for vertical Manchu word structure
- Asymmetric shrink: stronger horizontal shrink and weaker vertical shrink for vertical text boxes

## Recognition

The recognizer is an official-style SVTR compact model:

- Patch embedding
- Local/global token mixers
- CTC head

Project-specific additions:

- DAB: diacritic-aware branch
- Cross-attention fusion for detail and main sequence features
- OTP: orthographic transition penalty using a character transition matrix

Recognition comparison addition:

- SVTRv2 + NRTR shares the SVTRv2 visual sequence between the existing CTC
  head and an auxiliary autoregressive Transformer decoder.
- Training uses `L_CTC + lambda_NRTR * L_NRTR` with teacher forcing.
- Validation, held-out testing, and inference use the CTC branch, preserving the
  original SVTRv2 decoding and metric protocol.

## Full OCR

The full OCR pipeline performs:

1. Page detection
2. Reading-order sorting
3. Word crop extraction
4. Word recognition
5. JSON result export

Current crop extraction uses axis-aligned bbox crops around predicted polygons. Perspective rectification can be added later if needed.
