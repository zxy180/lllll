from character_attack import get_vulnerable_positions
import tools
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from DGD.help_functions import last_token_pool, get_relevant_documents, get_neighbor_ids, user_prompt_generation, get_embedding, check_success
import itertools
import random

def generate_random_modified_sentences(S, V, target_positions, k=1, random_char_num=2):
    random.seed(42)
    
    spaces = '_' * k
    extended_str = ''.join([spaces + c for c in S]) + spaces
    init_extended = list(extended_str)
    extended_length = len(init_extended)
    
    init_mask = []
    for _ in range(len(S)):
        init_mask += [0] * k + [1]
    init_mask += [0] * k
    assert len(init_mask) == extended_length, "扩展句与掩码长度不一致"

    vocab_indices = list(range(len(V)))
    random_char_num = min(random_char_num, len(vocab_indices))
    
    pos_char_candidates = []
    for _ in target_positions:
        random_chars = random.sample(vocab_indices, random_char_num)
        pos_char_candidates.append(random_chars)

    all_extended = []
    all_shrunk = []
    all_masks = []

    def generate_combinations(candidates, current=[]):
        if not candidates:
            yield current
            return
        for char_idx in candidates[0]:
            yield from generate_combinations(candidates[1:], current + [char_idx])

    for char_combo in generate_combinations(pos_char_candidates):
        new_extended = init_extended.copy()
        new_mask = init_mask.copy()

        for pos_idx, vocab_idx in enumerate(char_combo):
            z = target_positions[pos_idx]
            if z < 0 or z >= extended_length:
                continue

            if V[vocab_idx] != -1:
                new_extended[z] = chr(V[vocab_idx])
                new_mask[z] = 1
            else:
                new_extended[z] = '_'
                new_mask[z] = 0

        all_extended.append(''.join(new_extended))
        all_masks.append(new_mask)
        
        shrunk_chars = [new_extended[i] for i in range(extended_length) if new_mask[i] == 1]
        all_shrunk.append(''.join(shrunk_chars))

    return all_extended, all_shrunk, all_masks

def generate_all_modified_sentences_fast(S, V, target_positions, k=1):
    spaces = '_' * k
    extended_str = ''.join([spaces + c for c in S]) + spaces
    init_extended = list(extended_str)
    extended_length = len(init_extended)
    
    init_mask = []
    for _ in range(len(S)):
        init_mask += [0] * k + [1]
    init_mask += [0] * k
    assert len(init_mask) == extended_length, "扩展句与掩码长度不一致"

    vocab_indices = list(range(len(V)))
    pos_char_candidates = [vocab_indices for _ in target_positions]

    all_extended = []
    all_shrunk = []
    all_masks = []

    def generate_combinations(candidates, current=[]):
        if not candidates:
            yield current
            return
        for char_idx in candidates[0]:
            yield from generate_combinations(candidates[1:], current + [char_idx])

    for char_combo in generate_combinations(pos_char_candidates):
        new_extended = init_extended.copy()
        new_mask = init_mask.copy()

        for pos_idx, vocab_idx in enumerate(char_combo):
            z = target_positions[pos_idx]
            if z < 0 or z >= extended_length:
                continue

            if V[vocab_idx] != -1:
                new_extended[z] = chr(V[vocab_idx])
                new_mask[z] = 1
            else:
                new_extended[z] = '_'
                new_mask[z] = 0

        all_extended.append(''.join(new_extended))
        all_masks.append(new_mask)
        
        shrunk_chars = [new_extended[i] for i in range(extended_length) if new_mask[i] == 1]
        all_shrunk.append(''.join(shrunk_chars))

    return all_extended, all_shrunk, all_masks

def generate_all_modified_sentences(S, V, target_positions, k=1, alternative=-1, mask_store=None):
    if k != 0:
        spaces = ''.join(['_' for _ in range(k)])
        xx = ''.join([spaces + s for s in S] + [spaces])
        init_extended = [c for c in xx]
        init_mask = []
        for _ in range(len(S)):
            init_mask += [0] * k + [1]
        init_mask += [0] * k
        extended_length = len(init_extended)
        assert len(init_mask) == extended_length, "扩展句与掩码长度不一致"
    else:
        init_mask = mask_store
        init_extended = [c for c in S]
        extended_length = len(init_extended)
        assert len(init_mask) == extended_length, "扩展句与掩码长度不一致"
        
    n_pos = len(target_positions)
    vocab_indices = range(len(V))
    all_combinations = itertools.product(vocab_indices, repeat=n_pos)

    all_extended = []
    all_shrunk = []
    all_masks = []

    for combo in all_combinations:
        new_extended = init_extended.copy()
        new_mask = init_mask.copy()

        for pos_idx, vocab_idx in enumerate(combo):
            z = target_positions[pos_idx]
            u = vocab_idx
            if z < 0 or z >= extended_length:
                continue

            if V[u] != -1:
                new_extended[z] = chr(V[u])
                new_mask[z] = 1
            else:
                new_extended[z] = '_'
                new_mask[z] = 0

        extended_sent = ''.join(new_extended)
        all_extended.append(extended_sent)
        all_masks.append(new_mask)

        shrunk_chars = []
        for idx in range(extended_length):
            if new_mask[idx] == 1:
                shrunk_chars.append(new_extended[idx])
        shrunk_sent = ''.join(shrunk_chars)
        all_shrunk.append(shrunk_sent)

    return all_extended, all_shrunk, all_masks
