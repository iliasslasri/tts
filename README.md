# TTS: WIP

## Audio Samples

| Training            | Original                                               | Reconstructed                                                    |
| ------------------- | ------------------------------------------------------ | ---------------------------------------------------------------- |
| First trainings     | [▶ Original](./training_journal/assets/original_0.wav) | [▶ Reconstructed](./training_journal/assets/reconstructed_0.wav) |
| 2025-11-29/03-27-25 | [▶ Original](./training_journal/assets/original_1.wav) | [▶ Reconstructed](./training_journal/assets/reconstructed_1.wav) |

## At this point, the generation of the tokens for the RVQ is done autoregressively one at a time, without any hierarchical autoregressive modeling, the tokens are then flattened to compute a MSE Loss. the next iteration will implement RQ-Transformer (temporal+depth transformers) for hierarchical autoregressive modeling

Pull Encodec repo:

```bash
git submodule update --init
```
