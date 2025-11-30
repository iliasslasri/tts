import torch
import torchaudio

from encodec.encodec import EncodecModel
from encodec.encodec.utils import convert_audio

# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined by the bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
model.set_target_bandwidth(6.0)

print(model)

# Load and pre-process the audio waveform
wav, sr = torchaudio.load("./record_iliass.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
# Each element in `encoded_frames` is a tuple (encoded, mask)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
print(codes.shape)  # torch.Size([1, 8, T])

from torchviz import make_dot

x = torch.randn(1, 1, 24000)
frames = model(wav)
print(frames.shape)
make_dot(encoded_frames, params=dict(model.named_parameters())).render(
    "encodec_graph", format="pdf"
)
