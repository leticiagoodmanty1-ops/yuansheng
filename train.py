# Kaggle environment configuration - must be at the top
import sys
sys.path.append('/kaggle/input/mydataset')

import os
os.environ["WANDB_MODE"] = "offline"

import argparse
import numpy as np
import torch
import ujson as json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
import wandb


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        
        # Initialize gradient scaler for mixed precision
        # Use torch.amp.GradScaler('cuda') as per warning/new pytorch versions
        scaler = torch.amp.GradScaler('cuda')
        
        # Create Kaggle output directory if it doesn't exist
        os.makedirs('/kaggle/working', exist_ok=True)
        
        for epoch in train_iterator:
            model.zero_grad()
            # Add tqdm progress bar for training (position=0, leave=True to avoid log spam)
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                       desc=f"Epoch {epoch+1}/{int(num_epoch)}", position=0, leave=True)
            
            for step, batch in pbar:
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                
                # Use native PyTorch AMP (latest syntax)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = outputs[0] / args.gradient_accumulation_steps
                
                # Scale loss and backward pass
                scaler.scale(loss).backward()
                
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        # Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Optimizer step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                wandb.log({"loss": loss.item()}, step=num_steps)
                
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        # 保存 dev 集的预测结果（而不是 test 集）
                        pred = report(args, model, dev_features)
                        # Save to Kaggle working directory
                        result_path = "/kaggle/working/dev_result.json"
                        with open(result_path, "w") as fh:
                            json.dump(pred, fh)
                        print(f"Saved dev predictions to {result_path}")
                        if args.save_path != "":
                            # Update save path to Kaggle working directory
                            save_path = os.path.join("/kaggle/working", os.path.basename(args.save_path))
                            torch.save(model.state_dict(), save_path)
            
            pbar.close()
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    
    # Add tqdm progress bar for evaluation (position=0, leave=False to avoid log spam)
    pbar = tqdm(dataloader, desc=f"Evaluating {tag}", position=0, leave=False)
    
    for batch in pbar:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    
    pbar.close()

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        results = official_evaluate(ans, args.data_dir)
        best_f1 = results['overall']['f1']
        best_f1_ign = results['overall']['f1_ign']
        intra_f1 = results['intra']['f1']
        inter_f1 = results['inter']['f1']
    else:
        best_f1 = 0
        best_f1_ign = 0
        intra_f1 = 0
        inter_f1 = 0
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_Intra_F1": intra_f1 * 100,
        tag + "_Inter_F1": inter_f1 * 100,
    }
    return best_f1, output


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    
    # Add tqdm progress bar for report generation (position=0, leave=False)
    pbar = tqdm(dataloader, desc="Generating report", position=0, leave=False)
    
    for batch in pbar:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    
    pbar.close()

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/kaggle/input/mydataset/dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="/kaggle/input/mydataset/model/bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")

    args, unknown = parser.parse_known_args()  # Use parse_known_args for Jupyter/Kaggle compatibility
    wandb.init(project="DocRED")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        # 保存 dev 集的预测结果（而不是 test 集）
        pred = report(args, model, dev_features)
        # Save to Kaggle working directory
        os.makedirs('/kaggle/working', exist_ok=True)
        result_path = "/kaggle/working/dev_result.json"
        with open(result_path, "w") as fh:
            json.dump(pred, fh)
        print(f"Saved dev predictions to {result_path}")


if __name__ == "__main__":
    main()
