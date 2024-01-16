from typing import Any
import torch
import transformers
from peft import LoraConfig, get_peft_model
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def encoder_wrapper(encoder):
    encoder_class = type(encoder)
    
    def token2compressed(self, tokens, train=False):
        if train:
            self.train()
            return self(tokens, output_hidden_states=True)['hidden_states'][-1][:, ::3, :]
        else:
            self.eval()
            with torch.no_grad():
                return self(tokens, output_hidden_states=True)['hidden_states'][-1][:, ::3, :]
            
    def token2compressedAllInfo(self, tokens, train=False):
        if train:
            self.train()
            return self(tokens, output_hidden_states=True)['hidden_states'][-1]
        else:
            self.eval()
            with torch.no_grad():
                return self(tokens, output_hidden_states=True)['hidden_states'][-1]
    
    encoder_class.token2compressedAllInfo = token2compressedAllInfo
    encoder_class.token2compressed = token2compressed
    
def decoder_wrapper(decoder):
    decoder_class = type(decoder)
    
    def compressed2logits(self, compressed_vectors, decode_len=None, teacher_tokens=None, train=False):
        """

        Args:
            compressed_vectors (_type_): _description_
            decode_len (_type_, optional): 根据长度来决定decode多少. Defaults to None.
            teacher_tokens (_type_, optional): 没有decode_len时, 用于teacher forcing训练. Defaults to None.
            train (bool, optional): _description_. Defaults to False.

        """
        if train:
            self.train()
            
            if decode_len is not None:
                final_out = None
                for i in range(decode_len):
                    if i == 0:
                        true_o = self(inputs_embeds=compressed_vectors).logits
                        final_out = true_o[:, -1:, :]
                    else:
                        embeds = self.model.get_input_embeddings()(final_out.argmax(-1))
                        decoder_input_embeddings = torch.cat((compressed_vectors, embeds), dim=1)
                        true_o = self(inputs_embeds=decoder_input_embeddings).logits
                        final_out = torch.cat((final_out, true_o[:, -1:, :]), dim=1)
                return final_out
                
            else:
                teacher_embeds = self.model.get_input_embeddings()(teacher_tokens[:, :-1])
                decoder_input_embeds = torch.cat((compressed_vectors, teacher_embeds), dim=1)
                logits = self(inputs_embeds=decoder_input_embeds).logits[:, compressed_vectors.shape[1]-1:, :]
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), teacher_tokens.reshape(-1), ignore_index=-1)
            return loss
        else:
            self.eval()
            with torch.no_grad():
                final_out = None
                for i in range(decode_len):
                    if i == 0:
                        true_o = self(inputs_embeds=compressed_vectors).logits
                        final_out = true_o[:, -1:, :]
                    else:
                        embeds = self.model.get_input_embeddings()(final_out.argmax(-1))
                        decoder_input_embeddings = torch.cat((compressed_vectors, embeds), dim=1)
                        true_o = self(inputs_embeds=decoder_input_embeddings).logits
                        final_out = torch.cat((final_out, true_o[:, -1:, :]), dim=1)
            
            return final_out
    decoder_class.compressed2logits = compressed2logits

def tokenizer_wrapper(tokenizer):
    tokenizer_class = type(tokenizer)
    
    def strs2tokens(self, strs):
        return self(strs, return_tensors='pt').input_ids.cuda()
    
    def tokens2strs(self, tokens):
        return self.batch_decode(tokens)
    
    def logits2strs(self, logits):
        return self.batch_decode(logits.argmax(-1))
    
    tokenizer_class.strs2tokens = strs2tokens
    tokenizer_class.tokens2strs = tokens2strs
    tokenizer_class.logits2strs = logits2strs

def load_untrained_encoder_decoder_tokenizer(encoder_config_path, 
                                             decoder_config_path, 
                                             tokenizer_config_path):
    config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config_path)
    encoder = AutoModelForCausalLM.from_pretrained(
        encoder_config_path,
        load_in_4bit=True,
        device_map='cuda:0',
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_config_path,
        load_in_4bit=True,
        device_map='cuda:0',
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    
    encoder = get_peft_model(encoder, config)
    decoder = get_peft_model(decoder, config)
    
    encoder_wrapper(encoder)
    decoder_wrapper(decoder)
    tokenizer_wrapper(tokenizer)
    
    embedding_size = decoder.model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        decoder.model.resize_token_embeddings(len(tokenizer))
        encoder.model.resize_token_embeddings(len(tokenizer))
        
    return encoder, decoder, tokenizer


if __name__ == '__main__':
    path = "mistralai/Mistral-7B-v0.1"
    path = "meta-llama/Llama-2-7b-hf"
    encoder, decoder, tokenizer = load_untrained_encoder_decoder_tokenizer(
        path,
        path,
        path,
    )
    print()