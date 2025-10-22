# TTS

Don't forget to pull the submodules:

```bash
git submodule update --init
```

# TODO

#### Papers:
- [ ] EnCodec (Meta 2022) → discrete audio tokens, quantization, decoder.
- [ ] VALL-E → Text-to-Speech as discrete token modeling (text → semantic tokens → codec tokens).

#### Data:
- [ ] Prepare paired (text, audio) dataset (LJSpeech, VCTK, or custom).
- [ ] Resample all audio to 24kHz.
- [ ] Normalize loudness, trim silence, split long recordings.
- [ ] Clean and normalize text; optionally phonemize.
- [ ] Split into train/validation/test sets.

#### Models:
- [ ] Prepare Encodec
- [ ] Text Tokenization
- [ ] Model Architecture with Encodec