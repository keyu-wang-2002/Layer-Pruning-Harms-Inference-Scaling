import time
import torch
import torch.nn as nn
from collections import defaultdict
import fnmatch
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

############################### Perplexity ########################################

def eval_ppl_wikitext2(args, model, tokenizer, device):
    dataset = "wikitext2"
    print(f"evaluating perplexity on {dataset}")

    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.config.max_position_embeddings, tokenizer=tokenizer 
    )

    with torch.no_grad():
        ppl_test_wikitext2 = eval_ppl_wikitext(args, model, testloader, 1, device)

    return ppl_test_wikitext2

def eval_ppl_wikitext(args, model, testenc, bs=1, device=None):
    testenc = testenc.input_ids

    model.config.use_cache = False 

    nsamples = testenc.numel() // model.config.max_position_embeddings

    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)

        inputs = testenc[:,(i * model.config.max_position_embeddings):(j * model.config.max_position_embeddings)].to(device)
        inputs = inputs.reshape(j-i, model.config.max_position_embeddings)

        lm_logits = model(inputs, use_cache=False).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.config.max_position_embeddings * (j-i)

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.config.max_position_embeddings))

    torch.cuda.empty_cache()

    return ppl.item()

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def _get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

from datasets import load_dataset
import random

def _get_wikitext103(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def _get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('allenai/c4', 'en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'en', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return _get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'wikitext103' in name:
        return _get_wikitext103(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return _get_c4(nsamples, seed, seqlen, tokenizer)
    
#####################################################################################


############################### Zero shot ########################################

from functools import partial

def wrapped_model(args, model, tokenizer, device):
    hf_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=model.config.max_position_embeddings,
        device=device
    )

    return hf_model


def eval_arc_easy(args, model, tokenizer, device):
    processed_model = wrapped_model(args, model, tokenizer, device)

    results = evaluator.simple_evaluate(
        model=processed_model,
        tasks=["arc_easy"],
        num_fewshot=0
    )

    return results

def eval_gsm8k_cot_8_shot(args, model, tokenizer, device):
    processed_model = wrapped_model(args, model, tokenizer, device)
    
    results = evaluator.simple_evaluate(
        model=processed_model,
        tasks=["my_gsm8k_cot_8_shot"]
    )

    return results


def eval_aime24_nofigures(args, model, tokenizer, device):
    processed_model = wrapped_model(args, model, tokenizer, device)
    
    results = evaluator.simple_evaluate(
        model=processed_model,
        tasks=["aime24_nofigures"],
        num_fewshot=0,
        batch_size= 'auto',  #args.batch_size, 
        log_samples=args.log_samples, 
        gen_kwargs={max_gen_toks=args.max_gen_toks,max_tokens_thinking=args.max_tokens_thinking}
    )

    return results







