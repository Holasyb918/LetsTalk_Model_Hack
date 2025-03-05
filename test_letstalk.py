import torch
import os

import torch

from modules.Letstalk_PixAlpha import PixArtTransformer2DModel

model = PixArtTransformer2DModel.from_config(
    "./xl-2-1024-ms/transformer"
)

unet_type = "2d"
if torch.cuda.is_available():
    local_rank = torch.device("cuda:1")
else:
    local_rank = torch.device("cpu")

weight_dtype = torch.float16
model = model.to(local_rank, dtype=weight_dtype)

bsz = 2
frame_num = 2
timesteps = torch.randint(
    0,
    1000,
    (bsz,),
    device=local_rank,
).unsqueeze(1).repeat(1, frame_num).flatten()
# print("timesteps", timesteps.shape, timesteps) ; exit(-1)
timesteps = timesteps.long()
print("timesteps", timesteps.shape)
noisy_latents = torch.randn(bsz * frame_num, 4, 64, 64).to(local_rank, dtype=weight_dtype)
# timesteps = torch.randint(0, 1000, (2,)).to(local_rank)
reference_image_latents = torch.randn(bsz, 4, 64, 64).to(
    local_rank, dtype=weight_dtype
)
audio_frame_embeddings = torch.randn(bsz * frame_num, 32, 768).to(
    local_rank, dtype=weight_dtype
)
# kps_images = torch.randn(2, 320, 24, 64, 64).to(local_rank, dtype=weight_dtype)
# kps_images = torch.randn(bsz, 3, frame, 512, 512).to(local_rank, dtype=weight_dtype)
out = model(
    hidden_states=noisy_latents,  # ref image latent
    timestep=timesteps,
    encoder_hidden_states=audio_frame_embeddings,  # audio latent
    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
    reference_image_latents=reference_image_latents,
).sample
print(out.shape)
