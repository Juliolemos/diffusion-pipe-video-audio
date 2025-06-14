import json
import torch
from torch.utils.data import DataLoader
from diffusion_pipe.datasets.audio_video_dataset import AudioVideoDataset
from diffusion_pipe.models.wan import WanPipeline
from diffusion_pipe.trainers.lora_av_trainer import LoRAAVTrainer
from accelerate import Accelerator

def main():
    accelerator = Accelerator()
    manifest = json.load(open("annotations/av_manifest.json"))
    dataset = AudioVideoDataset(manifest)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = WanPipeline.from_pretrained("tdrussell/wan2.1").cuda()
    trainer = LoRAAVTrainer(model, dataloader, accelerator)
    trainer.train(num_epochs=10)

    model.save_pretrained("output/lora-av-model")

if __name__ == "__main__":
    main()
