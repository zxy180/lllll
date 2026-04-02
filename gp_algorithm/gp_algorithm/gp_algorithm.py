import torch
import torch.nn.functional as F
from torch import Tensor
import gpytorch
import random
import numpy as np
import scipy.stats as stats
import csv
import io
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from DGD.help_functions import last_token_pool, get_relevant_documents, get_neighbor_ids, user_prompt_generation, get_embedding, check_success, get_detailed_instruct, get_target_doc_indices_by_model
from character_attack import generate_sentece_bba, get_vulnerable_positions
import tools
from search_range import generate_all_modified_sentences, generate_all_modified_sentences_fast
import torch.distributions as dist
import json
from transformers import AutoTokenizer, AutoModel, LlamaTokenizerFast, MistralModel, AutoModelForCausalLM
import datetime
from RGA_model import SimpleGPModel, get_position_importance_with_all_gaps

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)

# ----------------------------------------------------------------------
# --- 0. 检索模型配置与加载 ---
# ----------------------------------------------------------------------
model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

local_model_path = "your_model_address"

try:
    tok = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModel.from_pretrained(local_model_path, torch_dtype=torch.float32).to(model_device)
except Exception as e:
    print(f"Warning: Could not load model/tokenizer. Using mock objects. Error: {e}")
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.ones(1, 5).long().to(device), 'attention_mask': torch.ones(1, 5).long().to(device)}
    class MockModel:
        def __init__(self):
            self.device = device
        def __call__(self, input_ids, attention_mask):
            class MockOutput:
                last_hidden_state = torch.rand(1, 5, 1024).to(self.device)
            return MockOutput()
    tok = MockTokenizer()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = MockModel()

# --- 1. CSV 数据和文档加载函数 ---
CSV_CONTENT = "Data/basketball_players.csv"
import torch
import csv
from tqdm import tqdm
from typing import List, Tuple

def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj

def load_and_process_documents(csv_content: str, dataset) -> Tuple[List[torch.Tensor], List[str]]:
    """
    加载 CSV 内容，构造文档，并生成嵌入。
    """
    nconst_embedding = []
    prompts = []
    max_length = 526
    counter = 0

    import csv
    
    tconst_title = {}
    print("正在加载电影标题数据 (Data/title.basics.tsv)...")
    
    try:
        with open("Data/title.basics.tsv", "r", encoding="utf-8") as f:
            rd_title = csv.reader(f, delimiter='\t')
            next(rd_title)
            for row in rd_title:
                tconst_title[row[0]] = row[2]
        print(f"✅ 电影标题加载完成，共 {len(tconst_title)} 条。")
    except FileNotFoundError:
        print("❌ 错误：在 Data 目录下找不到 title.basics.tsv 文件！")
        exit()
    except Exception as e:
        print(f"❌ 读取标题文件出错: {e}")
        exit()

    try:
        with open(csv_content, mode='r', encoding='utf-8') as fd:
            rd = csv.reader(fd)
            counter = 0
            
            for row in tqdm(rd, desc="Processing Documents", unit="row"):
                if counter > 1000:
                    break
                
                if counter > 0:
                    prompt = ""
                    object_name = None
                    
                    if dataset == "IMDB_name.basics":
                        prompt = row[1] + " was born in " + row[2] + ", " + "and died in " + row[3] + ". He/She's primary professions are " + ', '.join(map(str, row[4].split(","))) + "."
                        tts = row[5].split(",")
                        prompt += " He/She is known for movies:"
                        for t in tts:
                            if t == tts[-1]:
                                prompt += " '" + tconst_title[t] + "'."
                            else:
                                prompt += " '" + tconst_title[t] + "',"
                        object_name = row[1]
                    
                    elif dataset == "nobel-prize-laureates":
                        lst = row[0].split(";")
                        if len(lst) < 10:
                            counter += 1
                            continue
                        
                        pronoun = "He" if lst[11] == "male" else "She"
                        
                        if len(lst) < 17:
                            prompt = f"{lst[1]} {lst[2]} was born in {lst[5]}, {lst[3]}. And died in {lst[6]}, {lst[4]}. {pronoun} won Nobel prize in {lst[13]}, {lst[12]}, {lst[15][3:-3]}."
                        else:
                            prompt = f"{lst[1]} {lst[2]} was born in {lst[5]}, {lst[3]}. And died in {lst[6]}, {lst[4]}. {pronoun} won Nobel prize in {lst[13]}, {lst[12]}, {lst[15][3:-3]}. {pronoun} work in {lst[16]}, {lst[17]} {lst[18]}."
                        
                        object_name = lst[1] + " " + lst[2]
                    
                    elif dataset == "basketball_players":
                        prompt = row[1] + ", born on " + row[2] + ", and played for " + row[3] + ". " + row[1] + " has been honored with the " + row[4] + "."
                        object_name = row[1]
                    
                    input_texts = prompt
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    batch_dict = tok(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(model_device)
                    
                    with torch.no_grad():
                        outputs = model(**batch_dict)
                        embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to("cpu")
                    
                    nconst_embedding.append(embedding[0])
                    prompts.append(prompt)
                
                counter += 1
    
    except FileNotFoundError:
        print(f"Error: Data file '{csv_content}.csv' not found. Please ensure the file exists.")
    except Exception as e:
        print(f"An error occurred during data processing: {e}")

    return nconst_embedding, prompts

def get_bo_score(query: str, target_doc_index: int = 0, topK: int = 5, y_best_emb=None) -> float:
    """
    检索评分函数，根据目标文档的排名计算 BO 评分。
    """
    top_k = get_relevant_documents(torch.stack(MOCK_PASSAGES_emb), query, topK, model, tok, y_best_emb)
    print("#########################当前查询的top-k#################################")
    print("top_k:", top_k)
    print("target_doc_index:", target_doc_index)
    
    if target_doc_index in top_k:
        rank = np.where(top_k == target_doc_index)[0][0]
        score = 100.0 / (rank + 1)
    else:
        score = 0.0

    score += random.uniform(-0.5, 0.5)
    return max(0, score), top_k

from qmapper_sfr_mini import QueryMapper
from RGA_model import SimpleGPModel
from torch.distributions.normal import Normal

def fit_gp_model(model, likelihood, train_x, train_y, training_iterations=100):
    """
    拟合高斯过程模型（通过最大化边缘对数似然）。
    """
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    print("\n--- 开始拟合 GP 代理模型 ---")
    for i in range(training_iterations):
        optimizer.zero_grad()
        with gpytorch.settings.cholesky_jitter(1e-2):
            output = model(train_x)
            loss = -mll(output, train_y)
        loss.backward()
        
        if (i + 1) % 10 == 0:
            print(f"迭代 {i+1} - 损失: {loss.item():.4f}")
        
        optimizer.step()
    print("--- 拟合完成 ---")
    return loss.item()

def predict_gp_model(model, likelihood, test_x):
    """
    使用拟合好的模型预测新点的均值和方差。
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        variance = observed_pred.variance
        return mean, variance

import torch
import gpytorch
from tqdm import tqdm

def sentence_similarity_last_token_pool_batch(
    sentence1: list,
    sentence2: str,
    ori_sentence,
    model,
    tokenizer,
    device=compute_device,
    batch_size: int = 1024
) -> tuple[list, list]:
    """
    批量计算列表中的每个句子与单个句子的相似度。
    """
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    model_device = next(model.parameters()).device
    model.eval()
    
    def get_embedding_batch(sentences):
        if not sentences:
            return torch.tensor([], device=model_device)
        sentences = [get_detailed_instruct(task, sent) for sent in sentences]
        batch_dict = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model_device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )
        return embeddings.float()
    
    emb2 = get_embedding_batch([sentence2])
    emb_ori = get_embedding_batch([ori_sentence])
    
    all_similarities = []
    all_embeddings = []
    num_sentences = len(sentence1)
    
    with tqdm(total=num_sentences, desc="计算相似度", unit="sentence", ncols=80) as pbar:
        for i in range(0, num_sentences, batch_size):
            batch_sentences = sentence1[i:i+batch_size]
            emb1_batch = get_embedding_batch(batch_sentences)
            
            batch_similarities = F.cosine_similarity(emb1_batch, emb2).cpu().tolist()
            batch_similarities_fan = F.cosine_similarity(emb1_batch, emb_ori).cpu().tolist()
            emb1_batch_cpu = emb1_batch.cpu()
            batch_sum = [a - 0.5 * b for a, b in zip(batch_similarities, batch_similarities_fan)]
            batch_embeddings = emb1_batch.cpu().unbind(dim=0)
            all_embeddings.extend(batch_embeddings)
            all_similarities.extend(batch_sum)
            
            del emb1_batch, emb1_batch_cpu, batch_embeddings
            if torch.cuda.is_available():
                torch.cuda.synchronize(model_device)
                torch.cuda.empty_cache()
            pbar.update(len(batch_sentences))
    
    if hasattr(emb2, 'device') and emb2.device.type == 'cuda':
        emb2_cpu = emb2.cpu()
        del emb2
        if torch.cuda.is_available():
            torch.cuda.synchronize(model_device)
            torch.cuda.empty_cache()
        emb2 = emb2_cpu
    
    return all_similarities, all_embeddings

def sentence_similarity_last_token_pool(sentence1: str, sentence2: str, ori_s, model, tokenizer, device=torch.device("cuda")) -> float:
    """
    计算两个句子的相似度。
    """
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    model_device = next(model.parameters()).device
    
    def get_embedding(text):
        batch_dict = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model_device)
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            ).float()
        return embeddings.squeeze(0)
    
    sentence1 = get_detailed_instruct(task, sentence1)
    emb1 = get_embedding(sentence1)
    emb2 = get_embedding(sentence2)
    ori = get_embedding(ori_s)
    
    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    fan_sim = F.cosine_similarity(emb1.unsqueeze(0), ori.unsqueeze(0)).item()
    sim = sim - 0.5 * fan_sim
    return sim, emb1.cpu()

def upper_confidence_bound(mean, var, beta=2.0, i=0):
    """
    UCB (Upper Confidence Bound) 采集函数
    """
    std = torch.sqrt(var)
    ucb_score = mean + beta * std
    return ucb_score

def calculate_query_similarity(query1, query2, model, tok, device):
    """
    计算两个查询之间的余弦相似度。
    """
    batch_dict1 = tok(query1, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs1 = model(**batch_dict1)
        embeddings1 = last_token_pool(
            outputs1.last_hidden_state,
            batch_dict1['attention_mask']
        ).cpu()
    query_embedding1 = F.normalize(embeddings1, p=2, dim=1)
    
    batch_dict2 = tok(query2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs2 = model(**batch_dict2)
        embeddings2 = last_token_pool(
            outputs2.last_hidden_state,
            batch_dict2['attention_mask']
        ).cpu()
    query_embedding2 = F.normalize(embeddings2, p=2, dim=1)
    
    similarity = F.cosine_similarity(query_embedding1, query_embedding2).item()
    return similarity

import datasets
V = set([-1])
dataset_v = datasets.load_dataset("glue", "sst2")
attack_set = dataset_v['validation']
for x in attack_set['sentence']:
    V = V.union([ord(y) for y in set(x)])
vocab_V = list(V)

char_to_idx = {v: i for i, v in enumerate(vocab_V)}
char_to_idx[-1] = len(char_to_idx)
char_to_idx[-2] = len(char_to_idx)

def query_to_extended_indices(query_str, target_length=None):
    """
    将查询字符串转换为扩展的整数索引序列。
    """
    indices = []
    for char in query_str:
        char_code = ord(char)
        if char_code in char_to_idx:
            indices.append(char_to_idx[char_code])
        else:
            indices.append(char_to_idx[-1])
    
    tensor = torch.tensor(indices, dtype=torch.long)
    
    if target_length is not None:
        current_len = len(tensor)
        target_extended_length = target_length
        if current_len != target_extended_length:
            print(current_len)
            print(target_extended_length)
            print(query_str)
            print(len(query_str))
            print("扩展失败！！！！！！！")
            import sys
            sys.exit()
    return tensor

import tools

if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow_hub as hub
    MOCK_PASSAGES_emb, Prompts = load_and_process_documents(CSV_CONTENT, "basketball_players")
    dataset = "basketball"
    query_mode = "for_objects"
    TARGET_DOC_INDEX = []
    TARGET_DOC_INDEX_static = get_target_doc_indices_by_model("basketball_players", "SFR-Mistral")
    TARGET_TEXTS = []
    TARGET_QUERIES = []
    ORI_DOC_INDEX = []
    FIXED_TARGET_RANK = 50
    NUM_SAMPLES = 200
    
    all_indices = list(range(len(Prompts)))
    random.seed(42)
    random.shuffle(all_indices)
    
    if len(all_indices) > NUM_SAMPLES:
        selected_indices = all_indices[:NUM_SAMPLES]
    else:
        selected_indices = all_indices
        print(f"⚠️ 数据总量不足 {NUM_SAMPLES}，使用全部 {len(all_indices)} 个样本。")
    print(f"最终选定的测试样本索引: {selected_indices[:10]} ... (共 {len(selected_indices)} 个)")
    
    count_processed = 0
    for rank, i in enumerate(selected_indices):
        count_processed += 1
        print(f"添加样本进度 {count_processed}/{len(selected_indices)} (Index: {i})")
        
        ORI_DOC_INDEX.append(i)
        object = Prompts[i].split(',')[0].strip()
        user_prompt = user_prompt_generation(Prompts[i], object, dataset, query_mode)
        TARGET_QUERIES.append(user_prompt)
        target_id = TARGET_DOC_INDEX_static[rank]
        TARGET_DOC_INDEX.append(target_id)
        TARGET_TEXTS.append(Prompts[target_id])
    
    MOCK_DOC_COUNT = len(MOCK_PASSAGES_emb)
    SUCCESS_SCORE = 0.7
    MAX_ITERATIONS = 6
    CANDIDATE_BATCH_SIZE = 80
    query_count = 0
    results_store_dct = {}
    
    if MOCK_DOC_COUNT <= 1:
        print("\n无法运行优化流程：文档集无效或为空 (需要至少 2 个文档)。")
    else:
        map_device = compute_device
        map_tokenizer = AutoTokenizer.from_pretrained("your_enbedding_model_address")
        map_model = AutoModel.from_pretrained(
            "your_enbedding_model_address",
            trust_remote_code=True
        ).to(map_device)
        MAPPER = QueryMapper(
            tokenizer=map_tokenizer,
            model=map_model,
            max_len=len(TARGET_QUERIES[0]),
            device=map_device
        )
    
    store_i = 0
    success_count = 0
    opt_score_success_count = 0
    
    for target_query, target_text, target_doc_index, ori_doc_index in zip(TARGET_QUERIES, TARGET_TEXTS, TARGET_DOC_INDEX, ORI_DOC_INDEX):
        print(f"处理样本进度 {store_i}/{len(TARGET_QUERIES)} (Index: {store_i})")
        seq_length = len(target_query)
        print(f"序列长度: {seq_length}")
        previous_location = []
        previous_top_2 = ""
        results_store_dct[store_i] = {}
        _, ori_top_k = get_bo_score(target_query, ori_doc_index, 10)
        
        print(f"\n--- 处理原始查询: {ori_doc_index} ---")
        print(f"\n--- 处理目标查询: {target_query[:20]} ---")
        
        results_store_dct[store_i]["user_query"] = target_query
        results_store_dct[store_i]["target_doc"] = target_text
        results_store_dct[store_i]["ori_doc"] = Prompts[ori_doc_index]
        results_store_dct[store_i]["target_doc_id"] = target_doc_index
        results_store_dct[store_i]["ori_doc_index"] = ori_doc_index
        results_store_dct[store_i]["ori_topk"] = ori_top_k
        results_store_dct[store_i]["ori_top3_doc"] = [Prompts[ori_top_k[i]] for i in range(3)]
        print(f"目标文档索引: {target_doc_index}")
        print(f"**特征维度 (Mapper Max Len): {MAPPER.max_len}**")
        
        long_query_base = target_query
        seq_length = 2 * len(long_query_base) + 1
        print("seq_length:", seq_length)
        
        print("\n--- 1. 生成初始历史评估数据 ---")
        
        vulnerable_indices = get_vulnerable_positions(
            model=model,
            tok=tok,
            orig_S=long_query_base,
            tools=tools,
            last_token_pool_func=last_token_pool,
            device=model_device,
            topN=80,
            batch_size=256
        )
        print("找到的重要位置：", vulnerable_indices)
        
        all_historical_queries = []
        all_historical_queries_combined = []
        all_historical_queries_mask = []
        
        print("开始生成所有候选句子...")
        for idx in vulnerable_indices:
            queries, queries_combined, queries_mask = generate_all_modified_sentences_fast(
                long_query_base, vocab_V, [idx]
            )[:3]
            all_historical_queries.extend(queries)
            all_historical_queries_combined.extend(queries_combined)
            all_historical_queries_mask.extend(queries_mask)
        
        total_candidates = len(all_historical_queries_combined)
        print(f"生成的候选句子总数: {total_candidates}")
        
        batch_size = 600
        all_scores = []
        all_embeddings_list = []
        
        print(f"开始分批计算 {total_candidates} 个句子的得分...")
        
        for i in range(0, total_candidates, batch_size):
            batch_queries = all_historical_queries_combined[i:i+batch_size]
            batch_scores, batch_embeddings = sentence_similarity_last_token_pool_batch(
                batch_queries,
                target_text,
                Prompts[ori_doc_index],
                model,
                tok,
                device=compute_device,
                batch_size=256
            )
            all_scores.extend(batch_scores)
            all_embeddings_list.extend(batch_embeddings)
            torch.cuda.empty_cache()
            print(f"已计算 {min(i+batch_size, total_candidates)}/{total_candidates} 个句子的得分")
        
        scores_array = np.array(all_scores)
        min_score = scores_array.min()
        if min_score < 0:
            scores_array = scores_array - min_score
        
        probabilities = scores_array / scores_array.sum()
        target_size = 400
        print(f"按得分概率采样 {target_size} 个句子...")
        
        if total_candidates > target_size:
            np.random.seed(42)
            sampled_indices = np.random.choice(
                total_candidates,
                size=target_size,
                replace=False,
                p=probabilities
            )
            historical_queries = [all_historical_queries[i] for i in sampled_indices]
            historical_queries_combined = [all_historical_queries_combined[i] for i in sampled_indices]
            historical_queries_mask = [all_historical_queries_mask[i] for i in sampled_indices]
            historical_scores = [all_scores[i] for i in sampled_indices]
            all_embeddings = [all_embeddings_list[i] for i in sampled_indices]
        else:
            historical_queries = all_historical_queries
            historical_queries_combined = all_historical_queries_combined
            historical_queries_mask = all_historical_queries_mask
            historical_scores = all_scores
            all_embeddings = all_embeddings_list
        
        print(f"采样完成，最终训练数据量: {len(historical_queries_combined)}")
        query_count += len(historical_scores)
        y_best = max(historical_scores)
        max_index = historical_scores.index(y_best)
        y_best_emb = all_embeddings[max_index]
        
        train_x_indices = torch.stack([
            query_to_extended_indices(q, seq_length) for q in historical_queries
        ]).to(compute_device).float()
        
        train_texts = historical_queries_combined.copy()
        train_y = torch.tensor(historical_scores, dtype=torch.float).to(compute_device)
        store_init_y = train_y.clone()
        train_x = MAPPER.queries_to_features(historical_queries_combined).to(compute_device)
        train_x = torch.nn.functional.normalize(train_x, p=2, dim=-1)
        print(f"多样本X归一化后范数均值：{train_x.norm(dim=-1).mean().item():.4f}")
        single_x_shape = train_x[0].shape
        print(f"每个x的形状：{single_x_shape}")
        
        Y_mean = train_y.mean()
        Y_std = train_y.std()
        train_y = (train_y - Y_mean) / Y_std
        print(f"多样本Y标准化后均值：{train_y.mean().item():.4f}，标准差：{train_y.std().item():.4f}")
        print(f"**初始历史数据点数: {len(historical_queries)}**")
        print(f"**初始最佳分数 Y_best: {y_best:.2f}**")
        
        iteration = 0
        check_index = set()
        
        while iteration < MAX_ITERATIONS:
            iteration += 1
            print(f"\n=======================================================")
            print(f"--- 迭代 {iteration}/{MAX_ITERATIONS}：当前最佳分数 {y_best:.3f} ---")
            print(f"=======================================================")
            
            print("1. 拟合 GP 模型以更新置信区间...")
            from gpytorch.constraints import Interval
            from gpytorch.priors import GammaPrior
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=Interval(0.01, 0.5),
                noise_prior=GammaPrior(2.0, 20.0)
            ).to(compute_device)
            
            combined_x = torch.cat([train_x, train_x_indices], dim=1)
            print(f"拼接后X的形状：{combined_x.shape}")
            
            model_GP = SimpleGPModel(
                train_x=combined_x,
                predict_mode=False,
                train_y=train_y,
                likelihood=likelihood,
                custom_device=compute_device,
            ).to(compute_device)
            train_loss = fit_gp_model(model_GP, likelihood, combined_x, train_y, training_iterations=400)
            
            best_query_index = torch.argmax(train_y).item()
            best_query = historical_queries[best_query_index]
            best_query_combined = historical_queries_combined[best_query_index]
            best_query_mask = historical_queries_mask[best_query_index]
            
            sorted_indices = torch.argsort(train_y, descending=True)
            print(f"2. 基于当前最佳查询 ('{best_query[:20]}...') 生成新的候选点...")
            print("   >>> 使用汉明核学习的长度尺度 <<<")
            lengthscale = model_GP.covar_module.get_hamming_lengthscale()
            
            vulnerable_indices = get_position_importance_with_all_gaps(
                lengthscale=lengthscale,
                topK=3,
                temperature=10.0,
                noise_level=0.005
            )
            DEBUG = False
            print("找到的重要位置：", vulnerable_indices)
            candidate_queries, candidate_queries_combined, candidate_queries_mask = generate_all_modified_sentences(
                best_query, vocab_V, vulnerable_indices, k=0, mask_store=best_query_mask
            )[:3]
            print(f"已生成 {len(candidate_queries)} 个新的候选查询")
            
            candidate_indices = torch.stack([
                query_to_extended_indices(q, seq_length) for q in candidate_queries
            ]).to(compute_device)
            test_x = MAPPER.queries_to_features(candidate_queries_combined).to(compute_device)
            test_x = torch.nn.functional.normalize(test_x, p=2, dim=-1)
            
            combined_x = torch.cat([test_x, candidate_indices], dim=1)
            model_GP.covar_module.predict_mode = True
            mu, variance = predict_gp_model(model_GP, likelihood, combined_x)
            sigma = torch.sqrt(variance)
            
            mu = mu * Y_std + Y_mean
            variance = variance * (Y_std ** 2)
            
            if train_loss <= 0.35 and train_loss >= 0.1:
                current_beta = 2.5
            elif train_loss < 0.1:
                current_beta = 2.0
            elif train_loss > 0.35 and train_loss <= 0.8:
                current_beta = 3.5
            else:
                current_beta = 15
            print(f"   >>> 使用 UCB 采集 (Beta: {current_beta:.4f}) <<<")
            EI = upper_confidence_bound(mu, variance, beta=current_beta, i=iteration+1)
            
            top_ei_values, top_ei_indices = torch.topk(EI, min(len(EI), CANDIDATE_BATCH_SIZE))
            next_batch_queries = [candidate_queries[i.item()] for i in top_ei_indices]
            next_batch_queries_combined = [candidate_queries_combined[i.item()] for i in top_ei_indices]
            next_batch_queries_mask = [candidate_queries_mask[i.item()] for i in top_ei_indices]
            next_batch_mu = mu[top_ei_indices]
            next_batch_sigma = sigma[top_ei_indices]
            
            print("\n3. 期望改进 (EI) 选出的下一批候选点:")
            print(f"{'EI Rank':<9} | {'EI Value':<10} | {'预测均值 (μ)':<12} | {'预测标准差 (σ)':<10} | {'Query (前 20 字符)'}")
            print("-" * 80)
            
            for rank, (query_to_eval, query_to_eval_combined, query_to_eval_mask, ei_val, m, s) in enumerate(zip(
                    next_batch_queries, next_batch_queries_combined,
                    next_batch_queries_mask, top_ei_values, next_batch_mu, next_batch_sigma)):
                new_score, new_score_emb = sentence_similarity_last_token_pool(
                    query_to_eval_combined, target_text, Prompts[ori_doc_index], model, tok
                )
                query_count += 1
                print(f"{rank+1:<9} | {ei_val.item():<10.3f} | {m.item():<12.3f} | {s.item():<10.3f} | {str(query_to_eval_combined)}... (Actual Score: {new_score:.3f})")
                
                historical_queries.append(query_to_eval)
                historical_queries_combined.append(query_to_eval_combined)
                historical_queries_mask.append(query_to_eval_mask)
                train_texts.append(query_to_eval)
                
                new_x_indices = query_to_extended_indices(query_to_eval, seq_length).unsqueeze(0).to(compute_device)
                new_x = MAPPER.query_to_feature(query_to_eval_combined).unsqueeze(0).to(compute_device)
                train_x_indices = torch.cat([train_x_indices, new_x_indices], dim=0)
                new_y = torch.tensor([new_score], dtype=torch.float).to(compute_device)
                new_x = torch.nn.functional.normalize(new_x, p=2, dim=-1)
                train_x = torch.cat([train_x, new_x], dim=0)
                store_init_y = torch.cat([store_init_y, new_y], dim=0)
                Y_mean = store_init_y.mean()
                Y_std = store_init_y.std()
                train_y = (store_init_y - Y_mean) / Y_std
                print(f"新Y标准化后均值：{train_y.mean().item():.4f}，标准差：{train_y.std().item():.4f}")
                
                if new_score > y_best:
                    y_best = new_score
                    y_best_emb = new_score_emb
                    print(f"   ********* 新的最佳分数 Y_best 已更新至 {y_best:.3f} *********")
            
            best_query_index = torch.argmax(train_y).item()
            best_query = historical_queries[best_query_index]
            best_query_combined = historical_queries_combined[best_query_index]
            y_best_emb = y_best_emb.detach().cpu()
            real_score, top_k = get_bo_score(best_query, target_doc_index, 10, y_best_emb)
            
            print(f"检索分数:{real_score}")
            print("---------------------------------------------------------------------------------------------")
            
            if check_success(target_doc_index, ori_doc_index, top_k) == 2:
                success_count += 1
                break
            
            del model_GP, likelihood
            torch.cuda.empty_cache()
        
        print("-" * 80)
        results_store_dct[store_i]["is_success"] = check_success(target_doc_index, ori_doc_index, top_k)
        results_store_dct[store_i]["query_count"] = query_count
        results_store_dct[store_i]["opt_topK"] = top_k
        results_store_dct[store_i]["character_query"] = best_query_combined
        results_store_dct[store_i]["opt_top3_doc"] = [Prompts[top_k[i]] for i in range(3)]
        
        sim = calculate_query_similarity(results_store_dct[store_i]["user_query"], results_store_dct[store_i]["character_query"], model, tok, model_device)
        results_store_dct[store_i]["similarity"] = sim
        results_store_dct[store_i]["success_count"] = success_count
        store_i += 1
        
        print("\n=======================================================")
        final_query_index = torch.argmax(train_y).item()
        final_query = historical_queries[final_query_index]
        
        with open(f'results_GP_search_infer_0_199_basketball_SFR_Mistral_18_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(convert_to_json_serializable(results_store_dct), f, ensure_ascii=False, indent=4)
        print(f"✅ 已实时保存第 {store_i} 个查询结果至 JSON 文件")
        
        print(f"最终最佳分数: {y_best:.2f}")
        print(f"最佳查询: '{final_query}'")
        print(f"总评估次数: {query_count}")
        print("=======================================================")
