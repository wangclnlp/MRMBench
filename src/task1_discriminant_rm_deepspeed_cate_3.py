import os
import math
import sys
import socket
import time

import numpy as np
from tqdm import  tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_scheduler,
    AutoModelForCausalLM
)
from transformers import SchedulerType
from transformers.models.llama import LlamaTokenizer
from transformers import AutoTokenizer
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer, load_state_dict_into_model
from ds_utils import get_train_ds_config
import json
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--train_datasets",
        type=str,
        help="",
        required=True,
    )
    parser.add_argument(
        "--test_datasets",
        type=str,
        help="",
        required=True,
    )
    parser.add_argument(
        '--freeze',
        action='store_true',
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=-1,
                        help="Save checkpoint in steps.")
    parser.add_argument("--eval_steps",
                        type=int,
                        default=-1,
                        help="Evaluation in steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--local-rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                    type=str,
                    default='fp16',
                    choices=['fp16', 'bf16'],
                    help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    ## Use sequence level reward or lexical level reward
    parser.add_argument('--reward_type',
                        type=str,
                        default='seq',
                        choices=['seq', 'lex'],
                        help='Use sequence level reward or lexical level reward.')

    ## use the annotated scores to training the reward model
    parser.add_argument('--gpt_annotated_score',
                        type=bool,
                        default=False,
                        help='Use GPT-annotated scores to train the reward model.')
    parser.add_argument('--add_error_sample',
                        type=bool,
                        default=False,
                        help='Add error sample to the rank loss.')
    parser.add_argument('--train_only_v_head',
                        type=bool,
                        default=False,
                        help="if only train v_head"
    )
    parser.add_argument('--head_dropout',
                        type=float,
                        default=0.0,
                        help='dropout in the head')
    parser.add_argument("--trained_checkpoint",
                        type=str,
                        default="",
                        help=""
    )    
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    ## Listwise input setting
    parser.add_argument("--num_input_data",
                        type=int,
                        default=2,
                        help="Total number of candidate LLMs.")
    # Use multiple processes when processing datasets
    parser.add_argument("--num_dataset_process",
                        type=int,
                        default=8,
                        help="Use multi-process when set greater than 1")

    parser.add_argument("--pre_trained_reward_model",
                        type=str,
                        default="nofile",
                        help="The path of the pretrained reward model.")
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


def print_args(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape)).item()
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

class RewardModel_Subtask(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model.rwtranrsformer
        self.tokenizer = tokenizer
        self.PAD_ID = self.tokenizer.pad_token_id
        self.classification_head = nn.Linear(self.config.n_embd, 3, bias=False)

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        labels=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        outputs = self.classification_head(hidden_states)
        bs = outputs.shape[0]
        predict_features = []
        for i in range(bs):
            input_id = input_ids[i]
            value = outputs[i]
            c_inds = (input_id == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
            predict_features.append(value[c_ind-1][:])
        predict_features = torch.stack(predict_features)
        criterion = nn.CrossEntropyLoss()
        # import pdb ;pdb.set_trace()
        loss = criterion(predict_features, labels)
        
        return {
            "loss":loss,
            "logits":predict_features,
            "labels":labels
        }



class RewardModel(nn.Module):
    def __init__(self, base_model, tokenizer, is_reward=False):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = 0

        try:
            self.dropout = self.config.dropout
        except:
            # don't find dropout attribute
            self.dropout = 0.0 

        if self.dropout != 0.0:
            self.Dropout = nn.Dropout(p=self.dropout)

        if hasattr(self.config, "word_embed_proj_dim") and is_reward: 
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        elif hasattr(self.config, "word_embed_proj_dim"):
            # use zero matrix for the critic model
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
            nn.init.zeros_(self.v_head.weight)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            if is_reward:
                self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
            else:
                # use zero matrix for the critic model
                self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
                nn.init.zeros_(self.v_head.weight)

        self.rwtranrsformer = base_model
        self.tokenizer = tokenizer
        self.PAD_ID = tokenizer.pad_token_id

        self.loss_function_with_gpt_score = torch.nn.MSELoss()

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()
 


def create_reward_model(model_name_or_path,
                        trained_checkpoint,
                        tokenizer,
                        num_padding_at_beginning=0,
                        zero_stage=0):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
 
    import time

    start = time.time()

    base_model = AutoModel.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True)

    end = time.time()
    try:
        if torch.distributed.get_rank() == 0:
            print(f"> Creating model from_config took {end - start} seconds")
    except:
        print(f"> Creating model from_config took {end - start} seconds")

    rm_model = RewardModel(
        base_model,
        tokenizer,
        is_reward=True)

    if os.path.exists(trained_checkpoint):
        # load_checkpoint
        load_checkpoint_file = os.path.join(trained_checkpoint,"pytorch_model.bin")
        model_ckpt_state_dict = torch.load(load_checkpoint_file, map_location='cpu')
        load_state_dict_into_model(
            rm_model,
            model_ckpt_state_dict,
            "",
            zero_stage=zero_stage
        )

    rm_model_subtask = RewardModel_Subtask(
        rm_model,
        tokenizer
    )

    del rm_model
    return rm_model_subtask

class TextDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length=512):
        scr =  open(dataset_path,"r",encoding="utf-8")
        datasets = json.load(scr)
        self.texts = [data["input"] + data["output"] for data in datasets]
        self.labels = [data["label"] for data in datasets]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        tokenized = self.tokenizer(text, add_special_tokens=True)  
        if len(tokenized["input_ids"]) > self.max_length:
            return None  

        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  
            "attention_mask": encoding["attention_mask"].squeeze(0),  
            "labels": torch.tensor(label, dtype=torch.long)
        }
    
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  
    if len(batch) == 0:
        return None 
    
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }



def main():
    args = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['deepspeed_multinode_launcher'] = 'standard' 
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['wall_clock_breakdown'] = False


    set_random_seed(args.seed)
    torch.distributed.barrier()

    print("load tokenizer………………………………")
    def tokenizer_need_extra_token_id():
        keyword_list = ['llama', 'ziya', 'gpt2', 'rm']
        for keyword in keyword_list:
            if keyword in args.model_name_or_path.lower():
                return True
        return False
    
    if "llama-3" in args.model_name_or_path.lower() or "llama_3" in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                    fast_tokenizer=False,
                                                    trust_remote_code=True)
        tokenizer.pad_token_id = 128002
        tokenizer.bos_token_id = 128000
        tokenizer.eos_token_id = 128009
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True        
        tokenizer.padding_side = 'right'

        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        from tokenizers import processors
        tokenizer._tokenizer.post_processor = processors.Sequence(
                                                [
                                                    processors.ByteLevel(trim_offsets=False),
                                                    processors.TemplateProcessing(
                                                        single=f"{bos}:0 $A:0 {eos}:0",
                                                        pair=f"{bos}:0 $A:0 {bos}:1 $B:1 {eos}:1",
                                                        special_tokens=[
                                                            (bos, tokenizer.bos_token_id),
                                                            (eos, tokenizer.eos_token_id),
                                                        ],
                                                    ),
                                                ]
                                            )

    elif "baichuan" in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                    fast_tokenizer=False,
                                                    trust_remote_code=True)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        tokenizer.padding_side = 'right'
        
    elif tokenizer_need_extra_token_id():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                   fast_tokenizer=False)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        tokenizer.padding_side = 'right'
    else:
        tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    print("load dataset………………………………")

    train_dataset = TextDataset(args.train_datasets, tokenizer, args.max_seq_len)
    test_dataset = TextDataset(args.test_datasets, tokenizer, args.max_seq_len)

    print("create dataloader………………………………")
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)


    print("load model………………………………")
    rm_model=create_reward_model(args.model_name_or_path,args.trained_checkpoint,tokenizer)
    print(rm_model)
    #freeze
    rm_model.model.requires_grad_(False)
    rm_model.classification_head.requires_grad_(True)


    def evaluation_accuracy(labels,logits):
        if labels.size(0) != logits.size(0):
            raise ValueError("labels and logits not match!")
        prob = F.softmax(logits,dim=1)
        predicted_labels = torch.argmax(prob,dim=1)
        correct_predictions = (labels == predicted_labels).sum().item() #correct number
        return correct_predictions

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        num_data = 0
        total_loss = 0.0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = to_device(batch, device)
            # from rlhf_llama.deepspeed_chat.training.utils import pdb ; pdb.set_trace()
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs["loss"].item()
                labels = outputs["labels"]
                logits = outputs["logits"]
            num_data = num_data + labels.shape[0]
            correct_predictions  = evaluation_accuracy(labels,logits)
            total_predictions = total_predictions + correct_predictions
            total_loss = total_loss + loss
        if step == 0:
            eval_loss = total_loss
        else:
            eval_loss = total_loss/step
        # import pdb; pdb.set_trace()
        return total_predictions/num_data,eval_loss

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    print("start deepspeed initizlize………………")
    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    # if args.gradient_checkpointing:
    #     rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)

    global_step = 0

    # acc, eval_loss = evaluation_reward(rm_model, eval_dataloader)
    # print_rank_0(f"Base acc: {acc}")

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        mean_loss = 0
        rm_model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            global_step += 1
            batch = to_device(batch, device)
            outputs = rm_model(**batch)
            loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            print_rank_0(f'step: {step} loss:{loss}, ',
                         args.global_rank)


            if (global_step + 1) % args.eval_steps == 0:
                acc, eval_loss = evaluation_reward(rm_model, eval_dataloader)
                print_rank_0(f"Eval/epoch: {epoch+1}, Eval/step: {global_step+1}, Eval/loss: {eval_loss}, Eval/acc: {acc}")


        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)

        rm_model.tput_timer.update_epoch_count()


if __name__ == "__main__":
    main()
