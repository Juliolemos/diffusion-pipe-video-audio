from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchvision.io import read_video
import torch

class AudioVideoDataset(Dataset):
    def __init__(self, manifest_list):
        self.entries = manifest_list
        self.mel_fn = MelSpectrogram(sample_rate=48000, n_fft=1024, hop_length=256, n_mels=80)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        video, _, _ = read_video(entry['video_path'])
        video = video.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

        audio, sr = torchaudio.load(entry.get("audio_path", entry["video_path"]))
        if sr != 48000:
            audio = torchaudio.functional.resample(audio, sr, 48000)

        mel = self.mel_fn(audio).log().clamp(min=1e-5)  # [1, 80, T]

        return {
            "video": video,
            "mel": mel.unsqueeze(0),  # [1, 1, 80, T]
            "caption": entry["caption"]
        }
