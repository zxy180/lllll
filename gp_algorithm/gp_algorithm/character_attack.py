import tools
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import warnings
from DGD.help_functions import get_embedding, perturb_sentence, rank_tokens_by_importance, get_initial_ids, user_prompt_generation, check_success, check_MSEloss, get_first_output_token, get_neighbor_ids, check_success, get_relevant_documents

warnings.filterwarnings('ignore', category=DeprecationWarning, message="__array__ implementation doesn't accept a copy keyword")

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_vulnerable_positions(model, tok, orig_S, tools, last_token_pool_func, device, topN=2, batch_size=32):
    """
    寻找句子中最脆弱（最敏感）的攻击位置。
    """
    SS = tools.generate_all_sentences(orig_S, [ord(' ')], None, 1, alternative=-1)
    
    orig_S_dict = tok(orig_S, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**orig_S_dict)
        orig_S_embedding = last_token_pool_func(
            outputs.last_hidden_state, 
            orig_S_dict['attention_mask']
        ).cpu()
    
    all_SS_embeddings = []
    
    for i in range(0, len(SS), batch_size):
        batch_sentences = SS[i: i + batch_size]
        SS_dict = tok(batch_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**SS_dict)
            batch_embeddings = last_token_pool_func(
                outputs.last_hidden_state, 
                SS_dict['attention_mask']
            ).cpu()
            all_SS_embeddings.append(batch_embeddings)
    
    all_SS_embeddings = torch.cat(all_SS_embeddings, dim=0)
    
    orig_S_emb_norm = F.normalize(orig_S_embedding, p=2, dim=1).numpy()
    all_SS_emb_norm = F.normalize(all_SS_embeddings, p=2, dim=1).numpy()
    
    logits = np.sum(orig_S_emb_norm * all_SS_emb_norm, axis=-1)
    
    top_n = logits.argsort()[:topN]
    print(len(top_n))
    return top_n

def iterative_attack(
    model, 
    tok, 
    start_sentence, 
    target_sentence, 
    vocab_V, 
    tools, 
    last_token_pool_func, 
    device, 
    k_steps=3,        
    topN=2,           
    batch_size=32,
    target_id=-1, 
    user_prompt_id=-1,
    nconst_embeddings=None,
    topK=-1
):
    """
    执行多轮迭代的对抗攻击。
    """
    curr_sentence = start_sentence
    print(f"🚀 开始攻击！最大编辑距离(轮数): {k_steps}")
    print(f"初始句子: {curr_sentence}")
    
    target_dict = tok(target_sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        target_out = model(**target_dict)
        target_embedding = last_token_pool_func(
            target_out.last_hidden_state, 
            target_dict['attention_mask']
        ).cpu()
    
    target_emb_norm = F.normalize(target_embedding, p=2, dim=1).numpy()
    if target_emb_norm.ndim == 1:
        target_emb_norm = target_emb_norm[None, :]
    
    query_count = 0
    check_index = set()
    
    for step in range(k_steps):
        print(f"\n[第 {step + 1} 轮/{k_steps}] (当前句子长度: {len(curr_sentence)})")
        
        vulnerable_indices = get_vulnerable_positions(
            model=model,
            tok=tok,
            orig_S=curr_sentence, 
            tools=tools,
            last_token_pool_func=last_token_pool_func,
            device=device,
            topN=3,
            batch_size=batch_size
        )
        
        print(len(check_index))
        print(f"  找到脆弱位置索引: {vulnerable_indices}")
        
        attack_candidates = tools.generate_all_sentences(
            curr_sentence, 
            vocab_V, 
            vulnerable_indices, 
            1
        )
        
        if len(attack_candidates) == 0:
            print("  ⚠️ 未生成任何有效候选，跳过本轮")
            continue
            
        print(f"  生成 {len(attack_candidates)} 个候选句子，开始计算相似度...")
        
        all_cand_embeddings = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(attack_candidates), batch_size), desc="  Embedding计算", leave=False)
        except ImportError:
            iterator = range(0, len(attack_candidates), batch_size)

        for i in iterator:
            batch = attack_candidates[i: i + batch_size]
            batch_dict = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                out = model(**batch_dict)
                emb = last_token_pool_func(
                    out.last_hidden_state, 
                    batch_dict['attention_mask']
                ).cpu()
                all_cand_embeddings.append(emb)
        
        if not all_cand_embeddings:
            continue
            
        all_cand_embeddings = torch.cat(all_cand_embeddings, dim=0)
        
        cand_emb_norm = F.normalize(all_cand_embeddings, p=2, dim=1).numpy()
        target_scores = np.sum(target_emb_norm * cand_emb_norm, axis=-1)
        
        best_idx = np.argmax(target_scores)
        best_score = target_scores[best_idx]
        best_sentence = attack_candidates[best_idx]
        
        top3_indices = np.argsort(target_scores)[-3:][::-1]
        print(f"  本轮 Top-3 候选:")
        for idx in top3_indices:
            print(f"    [{target_scores[idx]:.4f}] {attack_candidates[idx]}")
        
        top_k = get_relevant_documents(torch.stack(nconst_embeddings), best_sentence, topK, model, tok)
        query_count = query_count + len(attack_candidates)
        
        success = check_success(target_id, user_prompt_id, top_k)
        if success == 2:
            return best_sentence, query_count, success, top_k
        
        curr_sentence = best_sentence
         
        if best_score > 0.98:
            print(f"  🎯 相似度已达 {best_score:.4f}，攻击成功，提前结束！")
            break

    print(f"\n✅ 攻击流程结束")
    print(f"原始: {start_sentence}")
    print(f"目标: {target_sentence}")
    print(f"最终: {curr_sentence}")
    
    return curr_sentence, query_count, success, top_k

def generate_sentece_bba(
    model, 
    tok, 
    start_sentence, 
    target_sentence,
    vocab_V, 
    tools, 
    last_token_pool_func, 
    device, 
    k_steps=3,        
    topN=2,           
    batch_size=32
):
    """
    执行多轮迭代的对抗攻击。
    """
    curr_sentence = start_sentence
    print(f"🚀 开始攻击！最大编辑距离(轮数): {k_steps}")
    print(f"初始句子: {curr_sentence}")
    
    target_dict = tok(target_sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        target_out = model(**target_dict)
        target_embedding = last_token_pool_func(
            target_out.last_hidden_state, 
            target_dict['attention_mask']
        ).cpu()
    
    target_emb_norm = F.normalize(target_embedding, p=2, dim=1).numpy()
    if target_emb_norm.ndim == 1:
        target_emb_norm = target_emb_norm[None, :]
    
    for step in range(k_steps):
        print(f"\n[第 {step + 1} 轮/{k_steps}] (当前句子长度: {len(curr_sentence)})")
        
        vulnerable_indices = get_vulnerable_positions(
            model=model,
            tok=tok,
            orig_S=curr_sentence, 
            tools=tools,
            last_token_pool_func=last_token_pool_func,
            device=device,
            topN=topN,
            batch_size=batch_size
        )
        
        vulnerable_indices_list = vulnerable_indices.tolist() if hasattr(vulnerable_indices, 'tolist') else list(vulnerable_indices)
        
        print(f"  找到脆弱位置索引: {vulnerable_indices_list}")
        
        attack_candidates = tools.generate_all_sentences(
            curr_sentence, 
            vocab_V, 
            vulnerable_indices_list, 
            1
        )
        
        print(f"  生成 {len(attack_candidates)} 个候选句子，开始计算相似度...")
        
        all_cand_embeddings = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(attack_candidates), batch_size), desc="  Embedding计算", leave=False)
        except ImportError:
            iterator = range(0, len(attack_candidates), batch_size)

        for i in iterator:
            batch = attack_candidates[i: i + batch_size]
            batch_dict = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                out = model(**batch_dict)
                emb = last_token_pool_func(
                    out.last_hidden_state, 
                    batch_dict['attention_mask']
                ).cpu()
                all_cand_embeddings.append(emb)
        
        if not all_cand_embeddings:
            continue
            
        all_cand_embeddings = torch.cat(all_cand_embeddings, dim=0)
        
        cand_emb_norm = F.normalize(all_cand_embeddings, p=2, dim=1).numpy()
        target_scores = np.sum(target_emb_norm * cand_emb_norm, axis=-1)
        
        best_idx = np.argmax(target_scores)
        best_score = target_scores[best_idx]
        best_sentence = attack_candidates[best_idx]
        
        top3_indices = np.argsort(target_scores)[-3:][::-1]
        print(f"  本轮 Top-3 候选:")
        for idx in top3_indices:
            print(f"    [{target_scores[idx]:.4f}] {attack_candidates[idx]}")
        
        if best_sentence == curr_sentence:
            print("  ⚠️ 警告：最佳句子与当前句子相同（陷入局部最优），停止攻击。")
            break
            
        curr_sentence = best_sentence
        
    return attack_candidates

if __name__ == "__main__":
    test_config = {
        "model": model,
        "tok": tok,
        "start_sentence": "hello world",
        "target_sentence": "attack!!",
        "vocab_V": V,
        "tools": tools,
        "last_token_pool_func": last_token_pool,
        "device": device,
        "k_steps": 10,
        "topN": 5,
        "batch_size": 8
    }
    
    print("开始测试对抗攻击...")
    result = iterative_attack(**test_config)
    
    print(f"\n测试完成!")
    print(f"起始句子: {test_config['start_sentence']}")
    print(f"目标句子: {test_config['target_sentence']}")
    print(f"最终句子: {result}")
