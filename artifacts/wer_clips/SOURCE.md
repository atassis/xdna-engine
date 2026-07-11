# WER clip set — source & provenance

## Dataset
- **FLEURS** (Few-shot Learning Evaluation of Universal Representations of Speech)
- HF hub: `google/fleurs` (ungated, public)
- License: **CC-BY-4.0**
- Paper: Conneau et al., 2022 (Google).

## How clips were obtained
- RU = config `ru_ru`, split `dev`. EN = config `en_us`, split `dev`.
- We range-streamed only the *head* of each split's `audio/<split>.tar.gz`
  (~14 MB RU / ~6 MB EN of the gz) and extracted whole tar entries — NO
  full-dataset download.
- Reference transcript = column 4 (0-indexed col 3) of `<split>.tsv`, the
  dataset's own normalized lowercased transcription. Stored lowercased in refs.json.
- Each WAV re-encoded to 16 kHz / mono / s16le via ffmpeg.

## Why FLEURS and not Common Voice
- Task preferred Mozilla Common Voice RU, but `mozilla-foundation/common_voice_17_0`
  is GATED: without an HF auth token the repo exposes only README + .gitattributes
  (0 data files), and the datasets-server returns nothing usable. No HF_TOKEN is
  present in this environment, so CV could not be pulled. FLEURS is the reputable,
  ungated RU+EN read-speech ASR benchmark used instead.

## Clips

| file | lang | split | source filename | duration |
|------|------|-------|-----------------|----------|
| ru_01.wav | ru_ru | dev | 10005687533826592442.wav | 13.7s |
| ru_02.wav | ru_ru | dev | 10042203520901241191.wav | 7.4s |
| ru_03.wav | ru_ru | dev | 10095490839809792710.wav | 11.8s |
| ru_04.wav | ru_ru | dev | 10121434821159649865.wav | 7.3s |
| ru_05.wav | ru_ru | dev | 10189896593749349276.wav | 12.1s |
| ru_06.wav | ru_ru | dev | 10215205936009281840.wav | 11.0s |
| ru_07.wav | ru_ru | dev | 10221250944584729444.wav | 4.8s |
| ru_08.wav | ru_ru | dev | 10267351211453560920.wav | 8.4s |
| ru_09.wav | ru_ru | dev | 10413922460366422025.wav | 10.3s |
| ru_10.wav | ru_ru | dev | 10419862324962592525.wav | 11.2s |
| ru_11.wav | ru_ru | dev | 10479463149729837036.wav | 11.0s |
| ru_12.wav | ru_ru | dev | 10479628937192443242.wav | 7.1s |
| ru_13.wav | ru_ru | dev | 1050167374088424092.wav | 10.3s |
| en_01.wav | en_us | dev | 10010138729160973689.wav | 6.5s |
| en_02.wav | en_us | dev | 1009709090964908274.wav | 7.9s |
| en_03.wav | en_us | dev | 10098964113747380446.wav | 4.1s |
| en_04.wav | en_us | dev | 10146705666908229607.wav | 4.9s |
