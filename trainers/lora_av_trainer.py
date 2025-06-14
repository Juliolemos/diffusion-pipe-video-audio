import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusion_pipe.utils.lora import inject_lora
from diffusion_pipe.modules.audio_encoder import SimpleAudioEncoder
from torch.nn.parallel import DistributedDataParallel as DDP

class LoRAAVTrainer:
    def __init__(self, model, dataloader, accelerator):
        self.model = model
        self.dataloader = dataloader
        self.accelerator = accelerator

        inject_lora(self.model)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
        self.audio_encoder = SimpleAudioEncoder().cuda()
        inject_lora(self.audio_encoder)

        self.model.train()
        self.text_encoder.eval()

        self.optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, self.model.parameters())) +
            list(filter(lambda p: p.requires_grad, self.audio_encoder.parameters())),
            lr=5e-5
        )

    def train_step(self, batch):
        video = batch["video"].cuda()
        mel = batch["mel"].cuda()
        tokens = self.tokenizer(batch["caption"], return_tensors="pt", padding=True, truncation=True).to("cuda")
        text_embed = self.text_encoder(**tokens).last_hidden_state
        audio_embed = self.audio_encoder(mel)

        # Combine embeddings
        combined_embed = text_embed + audio_embed.unsqueeze(1)  # [B, 1, D] + [B, T, D]

        # Call model loss
        noise = torch.randn_like(video)
        loss = self.model.train_step(video, noise, encoder_hidden_states=combined_embed, return_loss=True)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for batch in self.dataloader:
                loss = self.train_step(batch)
                print(f"[{epoch}] loss: {loss:.4f}")
