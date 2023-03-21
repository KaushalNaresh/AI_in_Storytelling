from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}

# Define a custom dataset class for text data
class MyTextDataset(Dataset):
    
    # Initialize the dataset with a tokenizer, file path, and block size
    def __init__(self, my_tokenizer, my_args, my_file_path='train', my_block_size=512):
        
        # Make sure the file path exists
        assert os.path.isfile(my_file_path)
        
        # Split the file path into directory and filename
        my_directory, my_filename = os.path.split(my_file_path)
        
        # Define the name of the cached features file
        my_cached_features_file = os.path.join(my_directory, my_args.model_name_or_path + '_cached_lm_' + str(my_block_size) + '_' + my_filename)

        # If the cached features file already exists and overwrite_cache is False, load the cached examples
        if os.path.exists(my_cached_features_file) and not my_args.overwrite_cache:
            with open(my_cached_features_file, 'rb') as my_handle:
                self.my_examples = pickle.load(my_handle)
        else:
            # Otherwise, tokenize the text data and build examples of length block_size
            self.my_examples = []
            with open(my_file_path, encoding="utf-8") as my_f:
                my_text = my_f.read()

            my_tokenized_text = my_tokenizer.convert_tokens_to_ids(my_tokenizer.tokenize(my_text))

            for i in range(0, len(my_tokenized_text)-my_block_size+1, my_block_size): 
                self.my_examples.append(my_tokenizer.build_inputs_with_special_tokens(my_tokenized_text[i:i+my_block_size]))
            
            # Cache the examples to speed up future runs
            with open(my_cached_features_file, 'wb') as my_handle:
                pickle.dump(self.my_examples, my_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Define the length of the dataset as the number of examples
    def __len__(self):
        return len(self.my_examples)

    # Get an example from the dataset at the given index
    def __getitem__(self, my_item):
        return torch.tensor(self.my_examples[my_item])


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = MyTextDataset(tokenizer, args, my_file_path=args.eval_data_file if evaluate else args.train_data_file, my_block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _rotate_checkpoints(my_args, my_checkpoint_prefix, my_use_mtime=False):
    # Check if save_total_limit is specified and greater than 0
    if not my_args.my_save_total_limit:
        return
    if my_args.my_save_total_limit <= 0:
        return

    # Get all the checkpoints with the specified prefix in the output directory
    my_glob_checkpoints = glob.glob(os.path.join(my_args.my_output_dir, '{}-*'.format(my_checkpoint_prefix)))

    # If the number of checkpoints is less than or equal to save_total_limit, return
    if len(my_glob_checkpoints) <= my_args.my_save_total_limit:
        return

    # Create a list of tuples with the mtime and checkpoint path (or checkpoint number) for each checkpoint
    my_ordering_and_checkpoint_path = []
    for my_path in my_glob_checkpoints:
        if my_use_mtime:
            # Use the mtime of the checkpoint file to determine its age
            my_ordering_and_checkpoint_path.append((os.path.getmtime(my_path), my_path))
        else:
            # Extract the checkpoint number from the checkpoint file name
            my_regex_match = re.match('.*{}-([0-9]+)'.format(my_checkpoint_prefix), my_path)
            if my_regex_match and my_regex_match.groups():
                my_ordering_and_checkpoint_path.append((int(my_regex_match.groups()[0]), my_path))

    # Sort the checkpoints by age (or checkpoint number)
    my_checkpoints_sorted = sorted(my_ordering_and_checkpoint_path)
    my_checkpoints_sorted = [my_checkpoint[1] for my_checkpoint in my_checkpoints_sorted]

    # Determine the number of checkpoints to delete and delete the oldest ones
    my_number_of_checkpoints_to_delete = max(0, len(my_checkpoints_sorted) - my_args.my_save_total_limit)
    my_checkpoints_to_be_deleted = my_checkpoints_sorted[:my_number_of_checkpoints_to_delete]
    for my_checkpoint in my_checkpoints_to_be_deleted:
        shutil.rmtree(my_checkpoint)


def mask_tokens(my_inputs, my_tokenizer, my_args):
    
    # Create labels from inputs
    my_labels = my_inputs.clone()
    
    # Create a probability matrix with mlm_probability for all non-special tokens
    my_probability_matrix = torch.full(my_labels.shape, my_args.mlm_probability)
    
    # Ignore special tokens that are not part of the actual input
    my_special_tokens_mask = [my_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in my_labels.tolist()]
    my_probability_matrix.masked_fill_(torch.tensor(my_special_tokens_mask, dtype=torch.bool), value=0.0)
    
    # Apply the mask
    my_masked_indices = torch.bernoulli(my_probability_matrix).bool()
    my_labels[~my_masked_indices] = -1  # We only compute loss on masked tokens

    # Fill the masked input with the mask token
    my_indices_replaced = torch.bernoulli(torch.full(my_labels.shape, 0.8)).bool() & my_masked_indices
    my_inputs[my_indices_replaced] = my_tokenizer.convert_tokens_to_ids(my_tokenizer.mask_token)

    # Fill the masked input with random tokens
    my_indices_random = torch.bernoulli(torch.full(my_labels.shape, 0.5)).bool() & my_masked_indices & ~my_indices_replaced
    my_random_words = torch.randint(len(my_tokenizer), my_labels.shape, dtype=torch.long)
    my_inputs[my_indices_random] = my_random_words[my_indices_random]

    return my_inputs, my_labels


def train(args, my_train_dataset, my_model, my_tokenizer):
    """Train the model."""

    # Calculate the training batch size based on the number of GPUs
    my_train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # Create a sampler to shuffle the training data
    my_train_sampler = RandomSampler(my_train_dataset) if args.local_rank == -1 else DistributedSampler(my_train_dataset)
    # Create a dataloader to load the training data
    my_train_dataloader = DataLoader(my_train_dataset, sampler=my_train_sampler, batch_size=my_train_batch_size)

    # Calculate the total number of training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(my_train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(my_train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    my_no_decay = ['bias', 'LayerNorm.weight']
    my_optimizer_grouped_parameters = [
        {'params': [p for n, p in my_model.named_parameters() if not any(nd in n for nd in my_no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in my_model.named_parameters() if any(nd in n for nd in my_no_decay)], 'weight_decay': 0.0}
        ]
    my_optimizer = AdamW(my_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    my_scheduler = get_linear_schedule_with_warmup(my_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')):
        # Load in optimizer and scheduler states
        my_optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
        my_scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))

    # Initialize training variables
    my_global_step = 0
    my_epochs_trained = 0
    my_steps_trained_in_current_epoch = 0
    if os.path.exists(args.model_name_or_path):
        my_global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
        my_epochs_trained = my_global_step // (len(my_train_dataloader) // args.gradient_accumulation_steps)
        my_steps_trained_in_current_epoch = my_global_step % (len(my_train_dataloader) // args.gradient_accumulation_steps)

    my_tr_loss = 0.0

    # Resize the model's token embeddings
    my_model_to_resize = my_model.module if hasattr(my_model, 'module') else my_model  
    my_model_to_resize.resize_token_embeddings(len(my_tokenizer))

    # Zero the gradients and initialize training iterator
    my_model.zero_grad()
    my_train_iterator = trange(my_epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Set random seed for reproducibility
    set_seed(args) 
    for _ in my_train_iterator:
        my_epoch_iterator = tqdm(my_train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for my_step, my_batch in enumerate(my_epoch_iterator):
            
            if my_steps_trained_in_current_epoch > 0:
                my_steps_trained_in_current_epoch -= 1
                continue

            my_inputs, my_labels = (my_batch, my_batch) # mask_tokens(batch, tokenizer, args) if args.mlm else
            my_inputs = my_inputs.to(args.device)
            my_labels = my_labels.to(args.device)
            my_model.train()
            my_outputs = my_model(my_inputs, labels=my_labels)
            my_loss = my_outputs[0]  

            if args.gradient_accumulation_steps > 1:
                my_loss = my_loss / args.gradient_accumulation_steps

            
            my_loss.backward()

            my_tr_loss += my_loss.item()
            if (my_step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(my_model.parameters(), args.max_grad_norm)
                my_optimizer.step()
                my_scheduler.step()  # Update learning rate schedule
                my_model.zero_grad()
                my_global_step += 1

                if args.local_rank in [-1, 0] and args.save_steps > 0 and my_global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    my_output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, my_global_step))
                    if not os.path.exists(my_output_dir):
                        os.makedirs(my_output_dir)
                    model_to_save = my_model.module if hasattr(my_model, 'module') else my_model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(my_output_dir)
                    my_tokenizer.save_pretrained(my_output_dir)

                    torch.save(args, os.path.join(my_output_dir, 'training_args.bin'))

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(my_optimizer.state_dict(), os.path.join(my_output_dir, 'optimizer.pt'))
                    torch.save(my_scheduler.state_dict(), os.path.join(my_output_dir, 'scheduler.pt'))

            if args.max_steps > 0 and my_global_step > args.max_steps:
                my_epoch_iterator.close()
                break
        if args.max_steps > 0 and my_global_step > args.max_steps:
            my_train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return my_global_step, my_tr_loss / my_global_step


def evaluate(my_args, my_model, my_tokenizer, my_prefix=""):

    my_eval_output_dir = my_args.my_output_dir

    my_eval_dataset = load_and_cache_examples(my_args, my_tokenizer, evaluate=True)

    if not os.path.exists(my_eval_output_dir) and my_args.my_local_rank in [-1, 0]:
        os.makedirs(my_eval_output_dir)

    my_args.my_eval_batch_size = my_args.my_per_gpu_eval_batch_size * max(1, my_args.my_n_gpu)
    # Note that DistributedSampler samples randomly
    my_eval_sampler = SequentialSampler(my_eval_dataset)
    my_eval_dataloader = DataLoader(my_eval_dataset, sampler=my_eval_sampler, batch_size=my_args.my_eval_batch_size)

    # multi-gpu evaluate
    if my_args.my_n_gpu > 1:
        my_model = torch.nn.DataParallel(my_model)

    my_eval_loss = 0.0
    my_nb_eval_steps = 0
    my_model.eval()

    for my_batch in tqdm(my_eval_dataloader, desc="Evaluating"):
        my_inputs, my_labels = mask_tokens(my_batch, my_tokenizer, my_args) if my_args.my_mlm else (my_batch, my_batch)
        my_inputs = my_inputs.to(my_args.my_device)
        my_labels = my_labels.to(my_args.my_device)

        with torch.no_grad():
            my_outputs = my_model(my_inputs, masked_lm_labels=my_labels) if my_args.my_mlm else my_model(my_inputs, labels=my_labels)
            my_lm_loss = my_outputs[0]
            my_eval_loss += my_lm_loss.mean().item()
        my_nb_eval_steps += 1

    my_eval_loss = my_eval_loss / my_nb_eval_steps
    my_perplexity = torch.exp(torch.tensor(my_eval_loss))

    my_result = {
        "perplexity": my_perplexity
    }

    my_output_eval_file = os.path.join(my_eval_output_dir, my_prefix, "eval_results.txt")
    with open(my_output_eval_file, "w") as my_writer:
        # logger.info("***** Eval results {} *****".format(my_prefix))
        for my_key in sorted(my_result.keys()):
            # logger.info("  %s = %s", my_key, str(my_result[my_key]))
            my_writer.write("%s = %s\n" % (my_key, str(my_result[my_key])))

    return my_result



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--block_size", default=-1, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str)

    parser.add_argument("--model_type", default="gpt2", type=str)
    parser.add_argument("--model_name_or_path", default="gpt2-base-cased", type=str)

    # parser.add_argument("--mlm", action='store_true')
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    # parser.add_argument("--config_name", default="", type=str)

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)

    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)

    # parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--server_ip', type=str, default='')
    parser.add_argument('--server_port', type=str, default='')
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument("--eval_all_checkpoints", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')

    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, _ = train(args, train_dataset, model, tokenizer)
        


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, my_prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
