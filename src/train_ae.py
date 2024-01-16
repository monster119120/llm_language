#!/usr/bin/env python
# coding=utf-8

import os
os.environ["WANDB_DISABLED"] = "true"

# 导入第三方库
import logging
import math
import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from tqdm.auto import tqdm
from transformers import (
    get_scheduler,
)
import transformers

# 导入我们的包
from args import parse_args, logger
from data import get_dataloader
from model import load_untrained_encoder_decoder_tokenizer, AverageMeter


def main():
    args = parse_args()

    accelerator = Accelerator(project_dir=args.output_dir)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if args.output_dir is None:
        raise NotImplementedError("No output_dir given!")    
    os.makedirs(args.output_dir, exist_ok=True)


    encoder, decoder, tokenizer = load_untrained_encoder_decoder_tokenizer(args.encoder_config_path, 
                                                                           args.decoder_config_path, 
                                                                           args.tokenizer_config_path)

    table_reduction = torch.nn.Linear(4096, int(os.environ.get('LENGTH')),device='cuda:0')
    table_increase = torch.nn.Linear(int(os.environ.get('LENGTH')), 4096,device='cuda:0')
    
    # 3.定义数据集
    train_dataloader, eval_dataloader = get_dataloader(args.dataset_name,
                                                        args.dataset_config_name, 
                                                        args.validation_split_percentage, 
                                                        args.block_size,
                                                        args.preprocessing_num_workers,
                                                        args.per_device_train_batch_size,
                                                        args.per_device_eval_batch_size,
                                                        tokenizer)

    # 4.注册一下模型，方便后续的存模型
    encoder, decoder, table_reduction, table_increase = accelerator.prepare(encoder, decoder, table_reduction, table_increase)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in decoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in decoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        { "params": table_reduction.parameters()},
        { "params": table_increase.parameters()},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_dataloader),
    )

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.num_train_epochs * len(train_dataloader)}")
    
    progress_bar = tqdm(range(args.num_train_epochs * len(train_dataloader)), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    L = int(os.environ.get('L'))
    def fsq(z):
        out1 = (L-1) * torch.sigmoid(z)
        with torch.no_grad():
            out2 = torch.round(out1) - out1
        return out1 + out2
    
    progress_bar.update(completed_steps)
    import random
    def random_substring(n, min_length = 2):
        valid_substrings = []

        weight_dict = {}
        for i in range(n - min_length + 1):
            for j in range(i + min_length, n + 1):
                if j-i not in weight_dict:
                    weight_dict[j-i] = 1
                else:
                    weight_dict[j-i] += 1
                valid_substrings.append((i, j))

        
        weights = [1/weight_dict[j - i] for i, j in valid_substrings]
        chosen_indices = random.choices(valid_substrings, weights=weights, k=3)
        
        return chosen_indices

    loss_meter = AverageMeter()

    # TODO 1 encoder后的token不缩短为n//3，而是直接进行VQ，然后decode

    # TODO 2 词表是多维的，比如用三个cls_head头，可能是三个[1-100]的index
    # embedding = embeds[10**6 x + 10**3 y + z]，词表最好也是三维的，怎么理解
    # [bs, seq_len, 2560]
    # [bs, seq_len//3, 2560]
    # 正常的词表embeds shape是[30000, 2560]
    
    
    # TODO 3 [bs, seq_len, 2560] -> [bs, seq_len, 10^6]
    
    for epoch in range(args.num_train_epochs):
        
        for step, batch in enumerate(train_dataloader):
            batch['input_ids'] = batch['input_ids'].to('cuda:0')
            random_indices = random_substring(batch['input_ids'].shape[1]//3)
            
            for start, end in random_indices:
                # compressed_vectors = encoder.token2compressed(batch['input_ids'], train=True)
                compressed_vectors = encoder.token2compressedAllInfo(batch['input_ids'], train=True)

                origin_dtype = compressed_vectors.dtype
                compressed_vectors = fsq(table_reduction(compressed_vectors.to(table_reduction.weight.dtype)))
                
                # compressed_vectors = table_reduction(compressed_vectors.to(table_reduction.weight.dtype))
                compressed_vectors = table_increase(compressed_vectors).to(origin_dtype)
                
                # input_tokens = batch['input_ids'][:, start*3:end*3]
                input_tokens = batch['input_ids'][:, start:end]
                sub_compressed_vectors = compressed_vectors[:, start:end]
                loss = decoder.compressed2logits(sub_compressed_vectors, teacher_tokens=input_tokens, train=True)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_meter.update(loss.detach().float())
            # progress_bar.set_description(f'loss: {loss.detach().float()}, loss_mean: {loss_meter.avg}')
            progress_bar.set_description("loss:{:.4f}, loss_mean:{:.4f}".format(loss.detach().float(), loss_meter.avg))
            progress_bar.update(1)
            completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    logger.info(f'测试可恢复性')
                    reconstructed_logits = decoder.compressed2logits(sub_compressed_vectors, decode_len=input_tokens.shape[1], train=False)
                    plaintext = tokenizer.tokens2strs(input_tokens)
                    reconstraction_plaintext = tokenizer.logits2strs(reconstructed_logits)
                    source_text = tokenizer.tokens2strs(batch['input_ids'])
                    logger.info(f'没裁剪原文{source_text[0]}')
                    logger.info(f'原文{plaintext[0]}')
                    logger.info(f'恢复{reconstraction_plaintext[0]}')

    accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}"))

if __name__ == "__main__":
    main()