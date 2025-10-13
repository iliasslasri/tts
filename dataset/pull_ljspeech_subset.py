import soundfile as sf
from datasets import load_dataset

dataset = load_dataset("MikhailT/lj-speech", split="full")

ds_head = dataset.take(2)
print(list(ds_head)[-1])