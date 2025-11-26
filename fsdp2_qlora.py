#!/usr/bin/env python3
"""
FSDP2 QLoRA Multi-GPU Training Script for Llama 3.1 8B

This script demonstrates QLoRA fine-tuning with PyTorch FSDP2 on multiple GPUs.
Compatible with Kaggle 2x Tesla T4 notebooks.

Usage:
    # Multi-GPU training (2+ GPUs)
    torchrun --nproc_per_node=2 fsdp2_qlora.py --max_steps 10
    
    # Single GPU baseline comparison  
    python fsdp2_qlora.py --max_steps 10 --single_gpu

Requirements:
    - torch >= 2.0.0
    - transformers
    - peft
    - bitsandbytes
    - accelerate
    - datasets
"""

import os
import sys
import argparse
import json
import time
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.api import ShardingStrategy

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        logger.info(f"Initialized distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return rank, local_rank, world_size
    else:
        # Single GPU mode
        return 0, 0, 1


def get_model_and_tokenizer(model_name, dtype=torch.float16):
    """Load quantized model and tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.device_count() == 1 else None,
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        torch_dtype=dtype,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_lora_config():
    """Configure LoRA parameters."""
    return LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def get_transformer_wrap_policy():
    """Define transformer block wrap policy for FSDP."""
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    return transformer_auto_wrap_policy(
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )


def setup_fsdp_config(model, use_cpu_offload=True, mixed_precision=True):
    """Configure FSDP settings."""
    # Mixed precision configuration
    mp_config = None
    if mixed_precision:
        mp_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    # CPU offload configuration
    cpu_offload = CPUOffload(offload_params=use_cpu_offload)
    
    # Auto wrap policy
    auto_wrap_policy = get_transformer_wrap_policy()
    
    # Sharding strategy
    sharding_strategy = ShardingStrategy.FULL_SHARD
    
    fsdp_config = {
        "sharding_strategy": sharding_strategy,
        "auto_wrap_policy": auto_wrap_policy,
        "cpu_offload": cpu_offload,
        "mixed_precision": mp_config,
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,
    }
    
    return fsdp_config


def apply_lora_and_prepare_model(model, lora_config, rank=0):
    """Apply LoRA adapters and prepare for training."""
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Freeze non-LoRA parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if rank == 0:
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def prepare_dataset(tokenizer, max_seq_length=2048):
    """Load and prepare training dataset."""
    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")
    
    def format_prompts(examples):
        """Format chat templates."""
        formatted_texts = []
        for text in examples["text"]:
            # Simple formatting - you can customize this
            formatted_text = f"Human: {text}\nAssistant: "
            formatted_texts.append(formatted_text)
        return {"text": formatted_texts}
    
    # Apply formatting
    dataset = dataset.map(
        format_prompts,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting prompts"
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_tensors=None,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def log_gpu_memory(rank, step, stage="training"):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.info(f"[Rank {rank}] Step {step} - {stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def train_single_gpu(model, tokenizer, dataset, args):
    """Single GPU baseline training."""
    logger.info("Starting single GPU baseline training...")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        fp16=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Log initial memory
    log_gpu_memory(0, 0, "start_single_gpu")
    
    # Train
    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time
    
    # Log final memory
    log_gpu_memory(0, args.max_steps, "end_single_gpu")
    
    logger.info(f"Single GPU training completed in {training_time:.2f} seconds")
    logger.info(f"Final training loss: {result.training_loss:.6f}")
    
    return result.training_loss, training_time


def train_multi_gpu_fsdp2(model, tokenizer, dataset, args):
    """Multi-GPU FSDP2 training."""
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    if rank == 0:
        logger.info(f"Starting FSDP2 training on {world_size} GPUs...")
    
    # Setup FSDP configuration
    fsdp_config = setup_fsdp_config(model)
    
    # Wrap model with FSDP
    model = FSDP(model, **fsdp_config)
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        fp16=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        # FSDP specific
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_forward_prefetch": False,
            "fsdp_use_orig_params": False,
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        },
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Log initial memory
    log_gpu_memory(rank, 0, "start_fsdp2")
    
    # Train
    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time
    
    # Log final memory
    log_gpu_memory(rank, args.max_steps, "end_fsdp2")
    
    if rank == 0:
        logger.info(f"FSDP2 training completed in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {result.training_loss:.6f}")
    
    # Gather loss from all ranks (should be identical)
    losses = [None] * world_size
    torch.distributed.all_gather_object(losses, result.training_loss)
    
    return result.training_loss, training_time, losses


def main():
    parser = argparse.ArgumentParser(description="FSDP2 QLoRA Training")
    parser.add_argument("--model_name", type=str, default="unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--single_gpu", action="store_true", help="Run single GPU baseline")
    parser.add_argument("--compare_with_single_gpu", action="store_true", help="Compare with single GPU baseline")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if not args.single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed + rank)
    
    try:
        # Load model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args.model_name)
        
        # Apply LoRA
        lora_config = get_lora_config()
        model = apply_lora_and_prepare_model(model, lora_config, rank)
        
        # Prepare dataset
        dataset = prepare_dataset(tokenizer, args.max_seq_length)
        
        if args.single_gpu or world_size == 1:
            # Single GPU training
            loss, time_taken = train_single_gpu(model, tokenizer, dataset, args)
            
            # Save results
            results = {
                "mode": "single_gpu",
                "final_loss": loss,
                "training_time": time_taken,
                "args": vars(args)
            }
            
        else:
            # Multi-GPU FSDP2 training
            loss, time_taken, all_losses = train_multi_gpu_fsdp2(model, tokenizer, dataset, args)
            
            # Save results (only on rank 0)
            if rank == 0:
                results = {
                    "mode": "fsdp2_multi_gpu",
                    "final_loss": loss,
                    "training_time": time_taken,
                    "all_rank_losses": all_losses,
                    "world_size": world_size,
                    "args": vars(args)
                }
                
                # Verify loss consistency across ranks
                loss_variance = max(all_losses) - min(all_losses)
                logger.info(f"Loss variance across ranks: {loss_variance:.6f}")
                if loss_variance > 1e-3:
                    logger.warning(f"High loss variance detected: {loss_variance:.6f} > 1e-3")
                else:
                    logger.info("✓ Loss consistency verified across all ranks")
        
        # Save results
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_dir}/training_results.json")
    
    finally:
        # Cleanup distributed training
        if world_size > 1:
            destroy_process_group()


if __name__ == "__main__":
    main()