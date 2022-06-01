#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Conditional text generation with GPT-2
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import string
import re

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, GPT2LMHeadModel

from yt_encoder import YTEncoder

import spacy

import pymorphy2


END_TAGS = {'<| endoftext|>', '<| endoftext |>', '<|endoftext|>', '<|endoftext |>'}
START_TAG = ' <|startoftext|>'

START_TAGS = {'<| startoftext|>', '<| startoftext |>', '<|startoftext|>', '<|startoftext |>'}

REMOVE_TAGS = {'<UNK>'}


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LENGTH = 120

FILTER_VALUE = -float('Inf')

NAME = 'NAME'
ORG = 'ORG'

suffix_len = 4

case_tag = 'Case='


case2pymorphy = {'Nom': 'nomn',
                 'Gen': 'gent',
                 'Acc': 'accs',
                 'Loc': 'loct',
                 'Dat': 'datv',
                 'Abl': 'ablt',
                 'Voc': 'voct'}


def postprocess(text):
    for symb in ',.?!"»':
        text = text.replace(' ' + symb, symb)

    for tag in REMOVE_TAGS:
        text = text.replace(tag, '')

    return text


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, 
                    device='cpu', max_input=1023, filter_single=[], filter_double=[]):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated[:,-max_input:]}
            outputs = model(**inputs)
            next_tokens = torch.zeros(num_samples, dtype=torch.long).to(device)
            for isample in range(num_samples):
                next_token_logits = outputs[0][isample, -1, :] / temperature

                next_token_logits[filter_single] = FILTER_VALUE
                # filter blank line = double \n
                if generated[isample, -1] in filter_double:
                    next_token_logits[generated[isample, -1]] = FILTER_VALUE

                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_tokens[isample] = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            sampled = next_tokens.unsqueeze(-1)
            generated = torch.cat((generated, sampled), dim=1)
    return generated


def generate(context_tokens, model, tokenizer, args):
    
    out = sample_sequence(
        model=model,
        context=context_tokens,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )
    out = out[0, len(context_tokens):].tolist()
    text = tokenizer.decode(out).strip()

    for tag in END_TAGS:
        if tag in text:
            text = text[:text.find(tag)]
            break

    for tag in START_TAGS:
        if text.startswith(tag):
            text = text[len(tag):].strip()
            break

    return text


def inflect_word(phrase, case, morpher):
    words = [x.strip(string.punctuation) for x in phrase.split()]
    result = []

    for word in words:
        try:
            word_inflected = morpher.parse(word)[0].inflect({'sing', case}).word
            if word.isupper():
                word_inflected = word_inflected.upper()
            elif word.istitle():
                word_inflected = word_inflected.title()
            result.append(word_inflected)
        except Exception:
            result.append(word)
    
    return ' '.join(result)


def replace_tags(text, tag, args, nlp, morpher):

    if tag not in text:
        return text

    doc = nlp(text)

    if tag == NAME:
        word = args.name
    elif tag == ORG:
        word = args.org
    else:
        raise ValueError('Invalid replacement')

    cases = []

    tag_node = None

    for token in doc:
        if token.text == tag:

            tag_node = token

    if tag_node is not None:

        for token in tag_node.children:

            case = None
            morph = str(token.morph)
            if case_tag in morph:
                case = morph[morph.find(case_tag) + len(case_tag):].strip()
                if '|' in case:
                    case = case[:case.find('|')].strip()

            case = case2pymorphy.get(case)

            cases.append(case)

    if not cases:
        cases = [None] * text.count(tag)
    
    parts = re.split(r'' + tag, text)

    parts_new = []

    while parts:
        parts_new.append(parts.pop(0))
        if not cases:
            break
        parts_new.append(inflect_word(word, cases.pop(0), morpher))

    text = ''.join(parts_new)

    text = postprocess(text)
        
    return text


def main(nlp, morpher):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                        help='Path to pre-trained model')
    parser.add_argument('--length', type=int, default=DEFAULT_LENGTH)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--no_cuda', action='store_true',
                        help='Avoid using CUDA when available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')

    parser.add_argument('--name', type=str, help='NAME value', required=False)
    parser.add_argument('--org', type=str, help='ORG value', required=False)
    
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    tokenizer = YTEncoder.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 

    context_tokens = tokenizer.encode(START_TAG)

    text = generate(context_tokens, model, tokenizer, args)

    logging.info('Generated text: ' + text)

    text = replace_tags(text, NAME, args, nlp, morpher)
    text = replace_tags(text, ORG, args, nlp, morpher)

    text = postprocess(text)

    logging.info('Resulting text: ' + text)

    return text


if __name__ == '__main__':

    
    nlp = spacy.load('ru_core_news_sm')

    morpher = pymorphy2.MorphAnalyzer()
    text = main(nlp, morpher)
