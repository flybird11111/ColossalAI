import argparse
import json
import os
import resource
from contextlib import nullcontext

import torch
from coati.dataset import DataCollatorForPreferenceDataset, StatefulDistributedSampler, load_tokenized_dataset
from coati.models import LoraConfig, convert_to_lora_module, disable_dropout
from coati.trainer import DPOTrainer
from coati.utils import load_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam

logger = get_dist_logger()


def train(args):
    lora_config = None
    if args.lora_config is not None:
        lora_config = LoraConfig.from_file(args.lora_config)
    # check lora compatibility
    if "gemini" in args.plugin and lora_config is not None and lora_config.r > 0:
        raise ValueError("LoRA is not supported in GeminiPlugin. Please use other plugin")

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "ddp":
        """
        Default torch ddp plugin without any acceleration, for
        debugging purpose acceleration, for debugging purpose
        """
        plugin = TorchDDPPlugin(find_unused_parameters=not args.grad_checkpoint)
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="static",
            initial_scale=2**16,
            max_norm=args.grad_clip,
            enable_gradient_accumulation=True,
            enable_flash_attention=args.use_flash_attn,
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            cpu_offload=True,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "3d":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=args.pp,
            sp_size=args.sp,
            sequence_parallelism_mode=args.sp_mode,
            zero_stage=args.zero_stage,
            enable_flash_attention=args.use_flash_attn,
            enable_sequence_parallelism=args.enable_sequence_parallelism,
            cpu_offload=True if args.zero_stage >= 1 and args.zero_cpu_offload else False,
            parallel_output=False,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
            microbatch_size=args.microbatch_size,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    ref_plugin = HybridParallelPlugin(
        tp_size=args.ref_tp,
        pp_size=1,
        zero_stage=args.zero_stage,
        enable_flash_attention=args.use_flash_attn,
        cpu_offload=True if args.zero_stage >= 1 and args.zero_cpu_offload else False,
        parallel_output=False,
        max_norm=args.grad_clip,
        precision=args.mixed_precision,
    )
    ref_booster = Booster(plugin=ref_plugin)

    init_ctx = nullcontext()
    with init_ctx:
        if args.use_flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrain,
                torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
                use_flash_attention_2=True,
            )
            coordinator.print_on_master(msg="Flash-attention enabled successfully")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.pretrain)

        if not args.disable_reference_model:
            if args.use_flash_attn:
                ref_model = AutoModelForCausalLM.from_pretrained(
                    args.pretrain,
                    torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
                    use_flash_attention_2=True,
                )
            else:
                ref_model = AutoModelForCausalLM.from_pretrained(args.pretrain)
        else:
            ref_model = None

        if args.lora_config is not None:
            model = convert_to_lora_module(model, lora_config=lora_config)
            for name, module in model.named_modules():
                if "norm" in name or "gate" in name:
                    module = module.to(torch.float32)
        disable_dropout(model)
        disable_dropout(ref_model)

    if args.grad_checkpoint:
        # Make sure gradient checkpointing can be activated.
        model.train()
        # Note, for some models, lora may not be compatible with gradient checkpointing.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        coordinator.print_on_master(msg="Gradient checkpointing enabled successfully")

    # configure tokenizer
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir is not None else args.pretrain
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, trust_remote_code=True)
    if hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
        try:
            # Some tokenizers doesn't allow to set pad_token mannually e.g., Qwen
            tokenizer.pad_token = tokenizer.eos_token
        except AttributeError as e:
            logger.warning(f"Unable to set pad token to eos token, {str(e)}")
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        logger.warning(
            "The tokenizer does not have a pad token which is required. May lead to unintended behavior in training, Please consider manually set them."
        )

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    # configure optimizer
    optim = HybridAdam(
        model_params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )

    # Configure dataset
    coordinator.print_on_master(f"Load dataset: {args.dataset}")
    mode_map = {"train": "train", "valid": "validation", "test": "test"}
    train_dataset = load_tokenized_dataset(dataset_paths=args.dataset, mode="train", mode_map=mode_map)
    data_collator = DataCollatorForPreferenceDataset(tokenizer=tokenizer, max_length=args.max_length)

    train_dataloader = plugin.prepare_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
        distributed_sampler_cls=StatefulDistributedSampler,
    )
    eval_dataloader = None
    if args.eval_dataset:
        eval_dataset = load_tokenized_dataset(dataset_paths=args.eval_dataset, mode="dev")
        eval_data_collator = DataCollatorForPreferenceDataset(tokenizer=tokenizer, max_length=args.max_length)

        eval_dataloader = plugin.prepare_dataloader(
            dataset=eval_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=eval_data_collator,
            distributed_sampler_cls=StatefulDistributedSampler,
        )
    else:
        logger.warning("No evaluation dataset is provided, skip evaluation")

    num_update_steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    if args.warmup_steps is None:
        args.warmup_steps = int(args.max_epochs * 0.025 * (len(train_dataloader) // args.accumulation_steps))
        coordinator.print_on_master(f"Warmup steps is set to {args.warmup_steps}")

    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optim,
        total_steps=args.max_epochs * num_update_steps_per_epoch,
        warmup_steps=args.warmup_steps,
        eta_min=0.1 * args.lr,
    )

    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)

    model, optim, _, train_dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        dataloader=train_dataloader,
    )
    ref_model, _, _, _, _ = ref_booster.boost(model=ref_model)

    torch.set_default_dtype(torch.float)

    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
    )

    start_epoch = 0
    sampler_start_idx = 0
    start_step = 0
    if args.checkpoint_path is not None:
        if "modeling" in args.checkpoint_path:
            coordinator.print_on_master(f"Continued pretrain from checkpoint {args.checkpoint_path}")
            booster.load_model(model, args.checkpoint_path)
        else:
            coordinator.print_on_master(f"Load model checkpoint from {args.checkpoint_path}")
            start_epoch, start_step, sampler_start_idx = load_checkpoint(
                load_dir=args.checkpoint_path,
                booster=booster,
                model=model,
                optimizer=optim,
                lr_scheduler=lr_scheduler,
            )
            assert isinstance(train_dataloader.sampler, StatefulDistributedSampler)
            train_dataloader.sampler.set_start_index(start_index=sampler_start_idx)

            coordinator.print_on_master(
                f"Loaded checkpoint {args.checkpoint_path} at epoch {start_epoch} step {start_step}"
            )
            coordinator.print_on_master(f"Loaded sample at index {sampler_start_idx}")

        coordinator.print_on_master(
            f"Checkpoint loaded max CUDA memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded CUDA memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
        )
        coordinator.print_on_master(
            f"Checkpoint loaded max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB"
        )

    trainer = DPOTrainer(
        actor=model,
        ref_model=ref_model,
        booster=booster,
        actor_optim=optim,
        plugin=plugin,
        actor_lr_scheduler=lr_scheduler,
        tokenizer=tokenizer,
        max_epochs=args.max_epochs,
        accumulation_steps=args.accumulation_steps,
        start_epoch=start_epoch,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        coordinator=coordinator,
        beta=args.beta,
        gamma=args.gamma,
        length_normalization=args.length_normalization,
        apply_loss_mask=not args.disable_loss_mask,
    )

    trainer.fit(
        train_preference_dataloader=train_dataloader,
        eval_preference_dataloader=eval_dataloader,
        log_dir=args.log_dir,
        use_wandb=args.use_wandb,
    )

    if lora_config is not None and lora_config.r > 0:
        # NOTE: set model to eval to merge LoRA weights
        model.eval()
    # save model checkpoint after fitting on only rank0
    if args.save_dir is not None:
        coordinator.print_on_master("Start saving final model checkpoint")
        booster.save_model(model, os.path.join(args.save_dir, "modeling"), shard=True)
        coordinator.print_on_master(
            f"Saved final model checkpoint at epoch {args.max_epochs} at folder {args.save_dir}"
        )

    coordinator.print_on_master(f"Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        choices=["gemini", "zero2", "zero2_cpu", "3d", "ddp"],
        help="Choose which plugin to use",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--loss_type", type=str, default="dpo_loss", help="dpo_loss or simpo_loss")
    parser.add_argument("--beta", type=float, default=0.1, help="beta in DPO loss")
    parser.add_argument("--gamma", type=float, default=0.0, help="gamma in SimPO loss")
    parser.add_argument("--length_normalization", default=False, action="store_true")
    parser.add_argument("--enable_sequence_parallelism", default=False, action="store_true")
    parser.add_argument("--zero_stage", type=int, default=0, help="Zero stage", choices=[0, 1, 2])
    parser.add_argument("--zero_cpu_offload", default=False, action="store_true")
    parser.add_argument("--sp_mode", type=str, default="split_gather", choices=["split_gather", "ring", "all_to_all"])
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--dataset", nargs="+", default=[])
    parser.add_argument("--eval_dataset", nargs="+", default=[])
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Checkpoint path if need to resume training form a checkpoint"
    )
    parser.add_argument("--config_file", type=str, default=None, help="Config file")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=2048, help="Model max length")
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--disable_loss_mask", default=False, action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--lora_config", type=str, default=None, help="low-rank adaptation config file path")
    parser.add_argument("--save_interval", type=int, default=1000, help="number of step between two checkpoints")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--grad_checkpoint", default=False, action="store_true")
    parser.add_argument("--use_flash_attn", default=False, action="store_true")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=2,
        help="Micro batch size for PP training. To activate PP training for DPO-like algorithm, you must keep size even and the size should be equal or greater than 2.",
    )
    # Parameter for reference model
    parser.add_argument(
        "--disable_reference_model",
        action="store_true",
        default=False,
        help="Disable the reference model (enabled by default)",
    )
    parser.add_argument(
        "--ref_tp",
        type=int,
        default=1,
        help="TP size for reference model; used only when reference model is too large.",
    )
    args = parser.parse_args()

    # fool proof hyperparameter setup
    if args.loss_type == "simpo_loss":
        args.length_normalization = True
        args.gamma = args.gamma if args.gamma > 0 else 1.4

    if args.config_file is not None:
        os.makedirs(os.path.dirname(args.config_file), exist_ok=True)
        with open(args.config_file, "w") as f:
            json.dump(args.__dict__, f, indent=4)
    train(args)
