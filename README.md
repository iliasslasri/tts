# TTS: WIP

## Listen to last checkpoint's output [tts_mode_out.wav](https://github.com/iliasslasri/tts/blob/main/reconstructed_model.wav).

## At this point, the generation of the tokens for the RVQ is done autoregressively one at a time, without any hierarchical autoregressive modeling, the tokens are then flattened to compute a MSE Loss. the next iteration will implement RQ-Transformer (temporal+depth transformers) for hierarchical autoregressive modeling

Don't forget to pull the submodules:

```bash
git submodule update --init
```
