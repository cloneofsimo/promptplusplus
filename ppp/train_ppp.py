# Secondly bootstrapped from:
# https://github.com/cloneofsimo/lora
# whatever man I wrote it anyways

# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
import itertools
import math
import os
import re
from typing import Optional, List, Literal

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler

from PIL import Image

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import wandb
import fire

from lora_diffusion import (
    save_all,
    prepare_clip_model_sets,
    evaluate_pipe,
)

from .dataset import PPPDataset, pp_extend
from .utils import PPPAttenProc


def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_tokens: List[str],
    initializer_tokens: List[str],
    device="cuda:0",
):

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    placeholder_token_ids = []

    for token, init_tok in zip(placeholder_tokens, initializer_tokens):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    unet.set_attn_processor(PPPAttenProc())

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids,
    )


@torch.no_grad()
def text2img_dataloader(
    train_dataset,
    train_batch_size,
    tokenizer,
    vae,
    text_encoder,
    cached_latents: bool = False,
):

    if cached_latents:
        cached_latents_dataset = []
        for idx in tqdm(range(len(train_dataset))):
            batch = train_dataset[idx]
            # rint(batch)
            latents = vae.encode(
                batch["instance_images"].unsqueeze(0).to(dtype=vae.dtype).to(vae.device)
            ).latent_dist.sample()
            latents = latents * 0.18215
            batch["instance_images"] = latents.squeeze(0)
            cached_latents_dataset.append(batch)

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids[0]},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    if cached_latents:

        train_dataloader = torch.utils.data.DataLoader(
            cached_latents_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        print("PTI : Using cached latent.")

    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    return train_dataloader


def loss_step(
    batch,
    unet,
    vae,
    text_encoder,
    scheduler,
    train_inpainting=False,
    t_mutliplier=1.0,
    mixed_precision=False,
    mask_temperature=1.0,
    cached_latents: bool = False,
    is_coarse_inversion: bool = False,
):
    weight_dtype = torch.float32
    if not cached_latents:
        latents = vae.encode(
            batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)
        ).latent_dist.sample()
        latents = latents * 0.18215

        if train_inpainting:
            masked_image_latents = vae.encode(
                batch["masked_image_values"].to(dtype=weight_dtype).to(unet.device)
            ).latent_dist.sample()
            masked_image_latents = masked_image_latents * 0.18215
            mask = F.interpolate(
                batch["mask_values"].to(dtype=weight_dtype).to(unet.device),
                scale_factor=1 / 8,
            )
    else:
        latents = batch["pixel_values"]

        if train_inpainting:
            masked_image_latents = batch["masked_image_latents"]
            mask = batch["mask_values"]

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    if train_inpainting:
        latent_model_input = torch.cat(
            [noisy_latents, mask, masked_image_latents], dim=1
        )
    else:
        latent_model_input = noisy_latents

    encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device))[0]

    if not is_coarse_inversion:
        _hs = {"this_idx": 0}

        for idx in range(16):
            _hs[f"CONTEXT_TENSOR_{idx}"] = encoder_hidden_states[idx : idx + 1, :, :]
    else:
        _hs = encoder_hidden_states

    model_pred = unet(latent_model_input, timesteps, _hs).sample

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    if batch.get("mask", None) is not None:

        mask = (
            batch["mask"]
            .to(model_pred.device)
            .reshape(
                model_pred.shape[0], 1, model_pred.shape[2] * 8, model_pred.shape[3] * 8
            )
        )
        # resize to match model_pred
        mask = F.interpolate(
            mask.float(),
            size=model_pred.shape[-2:],
            mode="nearest",
        )

        mask = (mask + 0.01).pow(mask_temperature)

        mask = mask / mask.max()

        model_pred = model_pred * mask

        target = target * mask

    loss = (
        F.mse_loss(model_pred.float(), target.float(), reduction="none")
        .mean([1, 2, 3])
        .mean()
    )

    return loss


def train_inversion(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps: int,
    scheduler,
    index_no_updates,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path: str,
    tokenizer,
    lr_scheduler,
    test_image_path: str,
    cached_latents: bool,
    accum_iter: int = 1,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    train_inpainting: bool = False,
    mixed_precision: bool = False,
    clip_ti_decay: bool = True,
    is_coarse_inversion: bool = False,
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    # Original Emb for TI
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        preped_clip = prepare_clip_model_sets()

    index_updates = ~index_no_updates
    loss_sum = 0.0

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        unet.eval()
        text_encoder.train()
        for batch in dataloader:

            lr_scheduler.step()

            with torch.set_grad_enabled(True):
                loss = (
                    loss_step(
                        batch,
                        unet,
                        vae,
                        text_encoder,
                        scheduler,
                        train_inpainting=train_inpainting,
                        mixed_precision=mixed_precision,
                        cached_latents=cached_latents,
                        is_coarse_inversion=is_coarse_inversion,
                    )
                    / accum_iter
                )

                loss.backward()
                loss_sum += loss.detach().item()

                if global_step % accum_iter == 0:
                    # print gradient of text encoder embedding
                    print(
                        text_encoder.get_input_embeddings()
                        .weight.grad[index_updates, :]
                        .norm(dim=-1)
                        .mean()
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():

                        # normalize embeddings
                        if clip_ti_decay:
                            pre_norm = (
                                text_encoder.get_input_embeddings()
                                .weight[index_updates, :]
                                .norm(dim=-1, keepdim=True)
                            )

                            lambda_ = min(1.0, 100 * lr_scheduler.get_last_lr()[0])
                            text_encoder.get_input_embeddings().weight[
                                index_updates
                            ] = F.normalize(
                                text_encoder.get_input_embeddings().weight[
                                    index_updates, :
                                ],
                                dim=-1,
                            ) * (
                                pre_norm + lambda_ * (0.4 - pre_norm)
                            )
                            print(pre_norm)

                        current_norm = (
                            text_encoder.get_input_embeddings()
                            .weight[index_updates, :]
                            .norm(dim=-1)
                        )

                        text_encoder.get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                        print(f"Current Norm : {current_norm}")

                global_step += 1
                progress_bar.update(1)

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if global_step % save_steps == 0:
                save_all(
                    unet=unet,
                    text_encoder=text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(
                        save_path, f"step_inv_{global_step}.safetensors"
                    ),
                    save_lora=False,
                )
                if log_wandb:
                    with torch.no_grad():
                        pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )

                        # open all images in test_image_path
                        images = []
                        for file in os.listdir(test_image_path):
                            if (
                                file.lower().endswith(".png")
                                or file.lower().endswith(".jpg")
                                or file.lower().endswith(".jpeg")
                            ):
                                images.append(
                                    Image.open(os.path.join(test_image_path, file))
                                )

                        wandb.log({"loss": loss_sum / save_steps})
                        loss_sum = 0.0
                        wandb.log(
                            evaluate_pipe(
                                pipe,
                                target_images=images,
                                class_token=class_token,
                                learnt_token="".join(placeholder_tokens),
                                n_test=wandb_log_prompt_cnt,
                                n_step=50,
                                clip_model_sets=preped_clip,
                            )
                        )

            if global_step >= num_steps:
                return


def train(
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    use_template: Literal[None, "object", "style"] = None,
    train_inpainting: bool = False,
    placeholder_tokens: str = "",
    preserve_prefix: bool = True,
    extend_amount: int = 1,
    do_coarse_inversion: bool = True,
    do_fine_inversion: bool = True,
    initializer_tokens: Optional[str] = None,
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    sample_batch_size: int = 1,
    max_train_steps_ti: int = 3000,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 2,
    gradient_checkpointing: bool = False,
    clip_ti_decay: bool = True,
    learning_rate_ti: float = 1e-3,
    cached_latents: bool = False,
    lr_scheduler: str = "linear",
    lr_warmup_steps: int = 0,
    weight_decay_ti: float = 0.00,
    use_8bit_adam: bool = False,
    device="cuda:0",
    extra_args: Optional[dict] = None,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "new_pti_project",
    wandb_entity: str = "new_pti_entity",
    proxy_token: str = "person",
    enable_xformers_memory_efficient_attention: bool = False,
):
    torch.manual_seed(seed)

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"steps_{max_train_steps_ti}_lr_{learning_rate_ti}_{instance_data_dir.split('/')[-1]}",
            reinit=True,
            config={
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    # print(placeholder_tokens, initializer_tokens)
    if len(placeholder_tokens) == 0:
        placeholder_tokens = []
        print("PTI : Placeholder Tokens not given, using null token")
    else:
        placeholder_tokens = placeholder_tokens.split("|")

        assert (
            sorted(placeholder_tokens) == placeholder_tokens
        ), f"Placeholder tokens should be sorted. Use something like {'|'.join(sorted(placeholder_tokens))}'"

    if extend_amount > 0:
        placeholder_tokens = placeholder_tokens[:1] + (
            pp_extend(
                placeholder_tokens[0],
                placeholder_tokens[0],
                preserve_prefix=False,
                extend_amount=extend_amount,
            )
        )

    if initializer_tokens is None:
        print("PTI : Initializer Tokens not given, doing random inits")
        initializer_tokens = ["<rand-0.017>"] * len(placeholder_tokens)
    else:
        initializer_tokens = initializer_tokens.split("|")

    assert len(initializer_tokens) == len(
        placeholder_tokens
    ), "Unequal Initializer token for Placeholder tokens."

    if proxy_token is not None:
        class_token = proxy_token
    class_token = "".join(initializer_tokens)

    print("PTI : Placeholder Tokens", placeholder_tokens)
    print("PTI : Initializer Tokens", initializer_tokens)

    # get the models
    text_encoder, vae, unet, tokenizer, placeholder_token_ids = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens,
        initializer_tokens,
        device=device,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    ti_lr = learning_rate_ti

    train_dataset = PPPDataset(
        instance_data_root=instance_data_dir,
        use_template=use_template,
        is_extended=False,
        main_token=placeholder_tokens[0],
        preserve_prefix=preserve_prefix,
        extend_amount=extend_amount,
        tokenizer=tokenizer,
        size=resolution,
        color_jitter=color_jitter,
        train_inpainting=train_inpainting,
    )

    train_dataset.blur_amount = 200

    train_dataloader = text2img_dataloader(
        train_dataset,
        train_batch_size,
        tokenizer,
        vae,
        text_encoder,
        cached_latents=cached_latents,
    )

    index_no_updates = torch.arange(len(tokenizer)) != -1

    for tok_id in placeholder_token_ids:
        index_no_updates[tok_id] = False

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    if cached_latents:
        vae = None

    ti_optimizer = optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=ti_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay_ti,
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=ti_optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps_ti,
    )

    # STEP 1 : Perform Course Inversion
    if do_coarse_inversion:

        train_dataset.is_extended = False

        train_inversion(
            unet,
            vae,
            text_encoder,
            train_dataloader,
            max_train_steps_ti,
            cached_latents=cached_latents,
            accum_iter=gradient_accumulation_steps,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            lr_scheduler=lr_scheduler,
            save_steps=save_steps,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_ids=placeholder_token_ids,
            save_path=output_dir,
            test_image_path=instance_data_dir,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=class_token,
            train_inpainting=train_inpainting,
            mixed_precision=False,
            tokenizer=tokenizer,
            clip_ti_decay=clip_ti_decay,
            is_coarse_inversion=True,
        )

    # STEP 2 : Perform Fine Inversion

    if do_fine_inversion:
        train_dataset.is_extended = True
        ti_optimizer.param_groups[0]["lr"] = ti_lr * 0.33

        train_inversion(
            unet,
            vae,
            text_encoder,
            train_dataloader,
            max_train_steps_ti,
            cached_latents=cached_latents,
            accum_iter=gradient_accumulation_steps,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            lr_scheduler=lr_scheduler,
            save_steps=save_steps,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_ids=placeholder_token_ids,
            save_path=output_dir,
            test_image_path=instance_data_dir,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=class_token,
            train_inpainting=train_inpainting,
            mixed_precision=False,
            tokenizer=tokenizer,
            clip_ti_decay=clip_ti_decay,
            is_coarse_inversion=False,
        )


def main():
    fire.Fire(train)
