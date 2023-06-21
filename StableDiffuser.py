# ref: https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/StableDiffuser.py

import torch
from diffusers import UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler


class StableDiffuser:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        scheduler="DDPM",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # VAE はいらない

        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device, dtype)

        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        ).to(device, dtype)

        if scheduler == "LMS":
            self.scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
        elif scheduler == "DDIM":
            self.scheduler = DDIMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )
        elif scheduler == "DDPM":
            self.scheduler = DDPMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )

        # self.eval()

    def get_noise(self, batch_size, img_size, generator=None):
        return (
            torch.randn(
                (
                    batch_size,
                    self.unet.config.in_channels,
                    img_size // 8,
                    img_size // 8,
                ),
                generator=generator,
            )
            .type(self.unet.dtype)
            .to(self.unet.device)
        )

    def text_tokenize(self, prompts):
        return self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def text_detokenize(self, tokens):
        return [
            self.tokenizer.decode(token)
            for token in tokens
            if token != self.tokenizer.vocab_size - 1
        ]

    def text_encode(self, tokens):
        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.unet.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts, generator=None):
        noise = self.get_noise(n_imgs, img_size, generator=generator).repeat(
            n_prompts, 1, 1, 1
        )

        latents = noise * self.scheduler.init_noise_sigma

        return latents

    def get_text_embeddings(self, prompts, n_imgs):
        text_tokens = self.text_tokenize(prompts)

        text_embeddings = self.text_encode(text_tokens)

        unconditional_tokens = self.text_tokenize([""] * len(prompts))

        unconditional_embeddings = self.text_encode(unconditional_tokens)

        text_embeddings = torch.cat(
            [unconditional_embeddings, text_embeddings]
        ).repeat_interleave(n_imgs, dim=0)

        return text_embeddings

    def predict_noise(
        self, iteration: int, latents, text_embeddings, guidance_scale=7.5, v_pred=False
    ):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latents = torch.cat([latents] * 2)

        latents = self.scheduler.scale_model_input(
            latents, self.scheduler.timesteps[iteration]
        )

        # predict the noise residual
        noise_prediction = self.unet(
            latents,
            self.scheduler.timesteps[iteration],
            encoder_hidden_states=text_embeddings,
        ).sample

        # TODO: v_parameterization?
        if v_pred:
            target = self.scheduler.get_velocity(latents, noise_prediction, iteration)
        else:
            target = noise_prediction

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = target.chunk(2)
        guided_target = noise_prediction_uncond + guidance_scale * (
            noise_prediction_text - noise_prediction_uncond
        )

        return guided_target

    @torch.no_grad()
    def diffusion(
        self,
        latents,
        text_embeddings,
        end_iteration=1000,
        start_iteration=0,
        return_steps=False,
        **kwargs
    ):
        latents_steps = []
        trace_steps = []

        for iteration in tqdm(range(start_iteration, end_iteration)):
            noise_pred = self.predict_noise(
                iteration, latents, text_embeddings, **kwargs
            )

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[iteration], latents
            )

            latents = output.prev_sample

            if return_steps or iteration == end_iteration - 1:
                if return_steps:
                    latents_steps.append(latents.cpu())
                else:
                    latents_steps.append(latents)

        return latents_steps, trace_steps
