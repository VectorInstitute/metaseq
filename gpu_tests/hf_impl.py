#!/usr/bin/env python3
import os
from transformers import OPTForCausalLM
import torch
from metaseq.hub_utils import tensorize_input, setup_vocab_and_merges


prompts = [
    "Today is a beautiful day and I want to",
# "In the city of",
# "Paris is the capital of France and",
# "Computers and mobile phones have taken",
]


model_path = os.path.join(os.path.dirname(__file__), "125m")
vocab_file, merges_file, tokenizer = setup_vocab_and_merges(model_path)


hf_model = OPTForCausalLM.from_pretrained(model_path)
hf_logits_list = []

with torch.no_grad():
    for prompt in prompts:
        input_ids = tensorize_input(tokenizer, prompt)
        # we know the HF stuff is correct, so use it as a reference
        input_ids[0][0] = 2
        logits_hf = hf_model(input_ids)[0]
        hf_logits_list.append(logits_hf.cpu())
        breakpoint()
