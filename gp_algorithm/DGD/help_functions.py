import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import faiss
import random
import os
import sys
import torch
import torch.nn.functional as F

device = torch.device("cuda")

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_embedding(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    # print("tokens: ", len(tokens["input_ids"][0]))
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
    # Assuming we're using the mean of the last hidden state as the embedding representation
    embedding = output.hidden_states[-1][0].mean(dim=0, keepdim=True)
    return embedding.to(device)

def perturb_sentence(sentence, index):
    words = sentence.split()
    if 0 <= index < len(words):
        words[index] = "[MASK]"
    return " ".join(words)

def rank_tokens_by_importance(sentence, model, tokenizer):
    original_embedding = get_embedding(sentence, model, tokenizer)
    distances = []
    
    for i in range(len(sentence.split())):
        perturbed = perturb_sentence(sentence, i)
        perturbed_embedding = get_embedding(perturbed, model, tokenizer)
        
        # Computing cosine distance
        # distance = cosine_distances(original_embedding, perturbed_embedding)[0][0]
        # Computing MSE distance
        distance = torch.nn.MSELoss()(original_embedding, perturbed_embedding).item()
        distances.append(distance)
        
    ranked_indices = np.argsort(distances)[::-1]
    words = sentence.split()
    ranked_words = [words[i] for i in ranked_indices]
    
    return ranked_words

def get_initial_ids(important_tokens, sentence, firstk, MODEL_NAME, model, tok):
    top_k_tokens = important_tokens[:firstk]
    
    # Sort them by their order in the original sentence
    sorted_top_k_tokens = sorted(top_k_tokens, key=lambda x: sentence.split().index(x))

    intital_sentence = ""
    for token in sorted_top_k_tokens:
        intital_sentence += (token + " ")

    if MODEL_NAME == "mistralai":
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"][:, 1:-1].to(device)[0]
    else:
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"].to(device)[0]
    return initial_ids[:firstk]

def get_initial_ids_logits(MODEL_NAME, user_prompt, desired_token, firstk, model, tok):
    name = user_prompt.split("was")[0].split("(")[0]
    if MODEL_NAME == "EleutherAI/gpt-j-6b":
        intital_sentence = name + desired_token[1:]
        print("intital_sentence: ", intital_sentence)
        
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"].to(device)[0].tolist()
        print("initial_ids length: ", len(initial_ids))
        if firstk > len(initial_ids):
            pad = initial_ids[-1]
            while len(initial_ids) < firstk:
                initial_ids.append(pad)
    elif MODEL_NAME == "mistralai":
        intital_sentence =  desired_token + name
        print("intital_sentence: ", intital_sentence)
        
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"].to(device)[0].tolist()[1:]
        print("initial_ids length: ", len(initial_ids))
        if firstk > len(initial_ids):
            while len(initial_ids) < firstk:
                initial_ids.append(918)
    else:
        intital_sentence = name + desired_token
        print("intital_sentence: ", intital_sentence)
        
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"].to(device)[0].tolist()[1:]
        print("initial_ids length: ", len(initial_ids))
        if firstk > len(initial_ids):
            while len(initial_ids) < firstk:
                initial_ids.append(918)
    
    return initial_ids[:firstk]

def user_prompt_generation(prompt_text, object, dataset, query_choose):
    if dataset == "imdb":
        if query_choose == "for_subjects":
            user_prompts = "According to IMDB dataset and your knowledges, who is known" + prompt_text.split("is known")[-1][:-1] + "?"
        elif query_choose == "for_objects":  
            surname = prompt_text.split()[1]
            lifespan = prompt_text.split(". ")[0].split(surname)[-1]
            movies = prompt_text.split(" is known for movies: ")[-1]
            user_prompts = "According to IMDB dataset and your knowledges, what movies " + prompt_text.split()[0] + " " + prompt_text.split()[1] + " has worked on and what were her/his roles?"
    elif dataset == "basketball":
        user_prompts = "According to wikidata (basketball players) dataset and your knowledges, what teams did " + object + " play for and what did she/he accomplish with them?"
    elif dataset == "book_query":
        user_prompts = "According to wikidata (book query) dataset and your knowledges, who wrote this book '" + object + "'?  And when was it published?"
    elif dataset == "Nobel_prize":
        user_prompts = "According to Nobel winner dataset and your knowledges, why and when did " + object + " win the Nobel prize?"
    return user_prompts


def check_success_2(user_prompt_id, target_prompt_id, prompt_text, token_ids, topk, model, tok, nconst_embedding):
    class SuppressStdout:
        def __enter__(self):
            self.original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self.original_stdout

    with SuppressStdout():
        with torch.no_grad():
            input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
            lst = token_ids + input_ids.tolist()[0]
            input_ids = torch.tensor([lst], dtype=torch.int).to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            data = np.array(nconst_embedding).astype('float32')
        
            # Generate some random queries
            query_vector = np.array(out.hidden_states[-1].to(torch.device('cpu')).tolist())
            queries = np.array([np.array(query_vector[0]).mean(axis=0).astype('float32')])
            
            dimension = 4096
            # Build the HNSW index in faiss
            index = faiss.IndexHNSWFlat(dimension, 16)  # 16 is the number of links per object
            index.hnsw.efConstruction = 40
            index.verbose = True
        
            faiss.normalize_L2(data)
            index.add(data)
    
            # Search using the index
            faiss.normalize_L2(queries)
            # index.hnsw.efSearch = 1000000
            distances, indices = index.search(queries, topk)
            
            Neighbors = []
            for j in range(topk):
                Neighbors.append(indices[0][j])
            
            # print("Neighbors: ", Neighbors)
            # print("user_prompt_id: ", user_prompt_id)
            # print("target_prompt_id: ", target_prompt_id)
            if user_prompt_id not in Neighbors:
                # if Neighbors[0] == target_prompt_id:
                #     return 2
                return 2
            elif user_prompt_id not in Neighbors:
                return -1
            else:
                return 0

def check_success(target_prompt_id, user_prompt_id, Neighbors):
    #print(target_prompt_id)
    #print(user_prompt_id)
    #print(Neighbors)
    if target_prompt_id in Neighbors:
        if Neighbors[0] == target_prompt_id and user_prompt_id not in Neighbors:
            return 2           #最好情况，第1名：target；ori不在topk
        elif Neighbors[0] == target_prompt_id and user_prompt_id in Neighbors:
            if Neighbors[1] == user_prompt_id:
                return 1       #第1名:target;第2名:ori
            else:
                return 0       #第1名:target;第2名不是ori
        else:
            return -2          #target在topk但不是第1名 
    else:
        return -1              #target不在top-k，失败 
        
  

def check_success_target_origin(user_prompt_id, target_prompt_id, prompt_text, token_ids, topk, model, tok, nconst_embedding):
    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        data = np.array(nconst_embedding).astype('float32')
    
        # Generate some random queries
        query_vector = np.array(out.hidden_states[-1].to(torch.device('cpu')).tolist())
        queries = np.array([np.array(query_vector[0]).mean(axis=0).astype('float32')])
        
        print("query shape: ", queries.shape)
        
        dimension = 4096
        # Build the HNSW index in faiss
        index = faiss.IndexHNSWFlat(dimension, 16)  # 16 is the number of links per object
        index.hnsw.efConstruction = 40
        index.verbose = True
    
        faiss.normalize_L2(data)
        index.add(data)

        # Search using the index
        faiss.normalize_L2(queries)
        # index.hnsw.efSearch = 1000000
        distances, indices = index.search(queries, topk)
        
        Neighbors = []
        for j in range(topk):
            Neighbors.append(indices[0][j])
        
        print("Neighbors: ", Neighbors)
        print("user_prompt_id: ", user_prompt_id)
        print("target_prompt_id: ", target_prompt_id)
        if target_prompt_id in Neighbors:
            if Neighbors[0] == target_prompt_id:
                return 2
            return 1
        else:
            return 0

def get_retrieved_result(user_prompt_id, prompt_text, topk, model, tok, nconst_embedding):
    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        data = np.array(nconst_embedding).astype('float32')
    
        query_vector = np.array(out.hidden_states[-1].to(torch.device('cpu')).tolist())
        queries = np.array([np.array(query_vector[0]).mean(axis=0).astype('float32')])
        
        print("query shape: ", queries.shape)
        
        dimension = 4096
        # Build the HNSW index in faiss
        index = faiss.IndexHNSWFlat(dimension, 16)  # 16 is the number of links per object
        index.hnsw.efConstruction = 40
        index.verbose = True
    
        faiss.normalize_L2(data)
        index.add(data)

        # Search using the index
        faiss.normalize_L2(queries)
        # index.hnsw.efSearch = 1000000
        distances, indices = index.search(queries, topk)
        
        
        # print(f"Query {user_prompt_id}:")
        Neighbors = []
        for j in range(topk):
            
            Neighbors.append(indices[0][j])
        
        return Neighbors

def construct_prompt(topk_passage_ids, prompts):
    composed_passage = ""
    for id in topk_passage_ids:
        composed_passage += prompts[id]
        composed_passage += " "
    return composed_passage

# def get_neighbor_ids(user_prompt_id, prompt_text, topk, model, tok, nconst_embedding):
#     with torch.no_grad():
#         input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
#         # input_ids = tok(prompt_text, padding='max_length', max_length=100, return_tensors="pt")["input_ids"].to(device)
#         lst = input_ids.tolist()[0]
#         input_ids = torch.tensor([lst], dtype=torch.int).to(device)
#         with torch.no_grad():
#             out = model(input_ids, output_hidden_states=True)
#         if isinstance(nconst_embedding, list):
#         # 如果是张量列表，先堆叠成一个大张量，再转到 CPU
#             data_tensor = torch.stack(nconst_embedding)
#         elif isinstance(nconst_embedding, torch.Tensor):
#         # 如果已经是张量，直接使用
#             data_tensor = nconst_embedding
#         else:
#         # 如果是 NumPy 数组或其他类型，直接尝试转换
#             data_tensor = torch.tensor(nconst_embedding)
        
#         data = np.array(nconst_embedding).astype('float32')
    
#         # Generate some random queries
#         query_vector = np.array(out.hidden_states[-1].to(torch.device('cpu')).tolist())
#         queries = np.array([np.array(query_vector[0]).mean(axis=0).astype('float32')])
        
#         #print("query shape: ", queries.shape)
        
#         dimension = 4096
        
#         index = faiss.IndexFlatL2(dimension)
#         index.add(data)
#         distances, indices = index.search(queries, topk)
#         target_id = indices[0][-1]  # Last 10 indices will be the farthest
        
#         return target_id
def get_neighbor_ids(user_prompt_id, prompt_text, model, tok, nconst_embedding, target_rank):
    """
    获取指定排名的邻居 ID。
    :param target_rank: 指定选取第几名的文档作为目标（默认20，确保在Top10之外）
    """
    
    # --- 1. 获取 Query Embedding (保持你原有的逻辑) ---
    with torch.no_grad():
        # 注意：这里去掉了 padding='max_length'，通常生成 embedding 不需要强制 padding，除非你的模型有特殊要求
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        
        # 你的原逻辑：处理 batch
        # input_ids = torch.tensor([input_ids.tolist()[0]], dtype=torch.int).to(device) 
        
        out = model(input_ids, output_hidden_states=True)
        
        # 获取 Embedding (Last token or Mean pooling - 保持你原有的 mean 逻辑)
        query_vector = out.hidden_states[-1].cpu().numpy()
        # 对 seq_len 维度求平均
        queries = np.array([query_vector[0].mean(axis=0)]).astype('float32')

    # --- 2. 准备 Faiss 索引 ---
    # 处理 nconst_embedding 数据格式
    if isinstance(nconst_embedding, list):
        if isinstance(nconst_embedding[0], torch.Tensor):
            data = torch.stack(nconst_embedding).cpu().numpy()
        else:
            data = np.array(nconst_embedding)
    elif isinstance(nconst_embedding, torch.Tensor):
        data = nconst_embedding.cpu().numpy()
    else:
        data = np.array(nconst_embedding)
    
    data = data.astype('float32')
    dimension = data.shape[1] # 自动获取维度 (例如 4096)

    index = faiss.IndexFlatL2(dimension)
    index.add(data)
    
    # --- 3. 核心修改：搜索并选取固定位置 ---
    # 我们搜索 target_rank + 5 个结果，以防万一
    # 例如：如果 target_rank=20，我们要取第 20 名，那至少得搜前 21 个
    search_k = target_rank + 5
    distances, indices = index.search(queries, search_k)
    
    # indices[0] 是一个列表，包含从近到远排序的 ID
    # indices[0][0] 是第1名 (Rank 1)
    # indices[0][9] 是第10名 (Rank 10)
    # indices[0][target_rank] 是第 target_rank + 1 名
    
    # 逻辑：我们要选一个固定的，且不在 Top 10 的
    # 直接取第 target_rank 个邻居
    candidate_id = indices[0][target_rank]
    
    # --- 4. 自身排重 (可选但推荐) ---
    # 如果检索出的文档正好是原文档自己 (index == user_prompt_id)，则顺延一位
    if candidate_id == user_prompt_id:
        candidate_id = indices[0][target_rank + 1]
        
    return candidate_id

def check_success_logits(prompt_text, token_ids, desired_token, original_token, model, tok):

    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            first_output_token_id = out.logits[0, -1].argmax().item()
            first_output_token = tok.decode(first_output_token_id).strip()

        print("new first_output_token: ", first_output_token)
        print("desired_token: ", desired_token)
        
        if first_output_token != original_token:
            if first_output_token == desired_token.strip():
                return 2, first_output_token
            return 1, first_output_token
        else:
            return 0, first_output_token

def check_success_logits_to_origin(prompt_text, token_ids, desired_token, original_token, model, tok):

    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            first_output_token_id = out.logits[0, -1].argmax().item()
            first_output_token = tok.decode(first_output_token_id).strip()

        print("new first_output_token: ", first_output_token)
        print("desired_token: ", desired_token)
        
        
        if first_output_token == desired_token.strip():
            return 1, first_output_token
        else:
            return 0, first_output_token
            

def check_success_logits_Instructed(Instructed_embeddings, prompt_text, token_ids, desired_token, original_token, model, tok):

    with torch.no_grad():
        instructed_ids = tok(Instructed_embeddings, return_tensors="pt")["input_ids"].to(device)
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = instructed_ids.tolist()[0] + token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            first_output_token_id = out.logits[0, -1].argmax().item()
            first_output_token = tok.decode(first_output_token_id).strip()

        print("new first_output_token: ", first_output_token)
        print("desired_token: ", desired_token)
        
        if first_output_token != original_token:
            if first_output_token == desired_token.strip():
                return 2, first_output_token
            return 1, first_output_token
        else:
            return 0, first_output_token

def check_MSEloss(target_prompt, prompt_text, token_ids, model, tok):
    
    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            user_tokens = out.hidden_states[-1] 
        user_emb = user_tokens[0].mean(dim=0, keepdim=True)

        input_ids = tok(target_prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            target_prompt_tokens = out.hidden_states[-1]
        
        target_emb = target_prompt_tokens[0].mean(dim=0, keepdim=True)
        
        mse_loss = loss_function(user_emb, target_emb).item()
    return mse_loss

def get_first_output_token(prompt, model, tok):
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        first_output_token_id = out.logits[0, -1].argmax().item()
        first_output_token = tok.decode(first_output_token_id)
    return first_output_token

# SFR mistral
def get_relevant_documents(nconst_embeddings, query, topK, model, tok,y_best_emb=None):
    model_device = next(model.parameters()).device

    if y_best_emb == None:
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        task_query = get_detailed_instruct(task, query)
        #print("task_query: ", task_query)
        batch_dict = tok(task_query, return_tensors="pt").to(model_device)
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state.to("cpu"), batch_dict['attention_mask'].to(torch.device('cpu'))).to("cpu")
        query_embedding = np.array(embeddings[0])
        #print(query_embedding.shape)
    else:
        query_embedding = y_best_emb
        #print(query_embedding.shape)
    
    queries = np.array([query_embedding])
    docs_embeddings = np.array(F.normalize(nconst_embeddings, p=2, dim=1))
    
    dimension = 4096
    # Build the HNSW index in faiss
    index = faiss.IndexHNSWFlat(dimension, 16)  # 16 is the number of links per object
    
    index.add(docs_embeddings)

    index.hnsw.efSearch = 1000000
    distances, indices = index.search(queries, topK)
    
    #print("top_k: ", indices[0])

    return indices[0]

def last_token_pool(last_hidden_states,
                 attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_target_doc_indices(dataset_name="basketball_players"):
    """
    返回指定数据集的目标文档索引列表
    
    Args:
        dataset_name: 数据集名称，支持 "basketball_players", "IMDB_name.basics", 
                      "books", "nobel-prize-laureates"
    
    Returns:
        list: 目标文档索引列表
    """
    
    # 篮球运动员数据集 (SFR-Mistral)
    basketball_indices = [721, 762, 145, 947, 765, 132, 397, 137, 791, 68, 594, 290, 448, 183, 591, 768, 137, 657, 762, 955, 637, 859, 873, 572, 92, 939, 640, 450, 863, 439, 448, 884, 223, 951, 976, 892, 563, 769, 631, 146, 506, 841, 653, 628, 146, 587, 835, 433, 863, 148, 859, 953, 976, 259, 860, 146, 110, 719, 458, 935, 130, 686, 719, 912, 139, 390, 768, 858, 937, 739, 534, 552, 727, 637, 458, 62, 995, 92, 976, 891, 413, 703, 873, 976, 769, 290, 884, 148, 290, 840, 490, 146, 873, 469, 65, 467, 525, 762, 891, 939, 450, 990, 727, 492, 625, 422, 884, 390, 211, 137, 211, 492, 884, 381, 458, 768, 663, 806, 772, 873, 592, 976, 835, 492, 835, 148, 886, 891, 824, 718, 313, 763, 829, 450, 995, 645, 706, 924, 995, 525, 390, 572, 390, 139, 976, 525, 492, 458, 679, 891, 703, 629, 249, 976, 132, 381, 703, 467, 492, 841, 452, 249, 313, 840, 92, 492, 525, 433, 886, 597, 718, 141, 259, 672, 504, 719, 840, 703, 513, 146, 886, 504, 58, 631, 62, 727, 62, 62, 552, 60, 769, 726, 858, 939, 838, 806, 223, 506, 637, 911]
    
    # IMDB 数据集 (SFR-Mistral)
    imdb_indices = [518, 815, 362, 744, 538, 688, 787, 853, 688, 371, 688, 744, 835, 691, 890, 381, 944, 883, 362, 176, 269, 712, 898, 399, 359, 815, 919, 757, 880, 826, 939, 567, 476, 394, 381, 560, 561, 937, 796, 498, 538, 541, 467, 95, 816, 953, 843, 682, 173, 399, 250, 691, 95, 837, 764, 853, 646, 481, 565, 391, 796, 399, 476, 749, 890, 775, 919, 853, 883, 757, 95, 502, 844, 380, 786, 291, 437, 682, 719, 796, 869, 814, 764, 891, 891, 362, 502, 467, 176, 843, 419, 853, 751, 282, 694, 541, 843, 998, 827, 538, 250, 853, 686, 846, 469, 469, 937, 744, 815, 545, 891, 757, 282, 891, 541, 853, 786, 837, 846, 751, 541, 757, 541, 519, 95, 399, 998, 673, 28, 560, 288, 541, 251, 399, 416, 883, 524, 519, 419, 744, 502, 419, 775, 541, 476, 82, 519, 381, 929, 846, 136, 787, 901, 691, 437, 971, 561, 444, 686, 458, 883, 370, 28, 437, 787, 816, 865, 883, 399, 853, 890, 173, 416, 890, 816, 726, 362, 476, 598, 173, 560, 281, 281, 310, 846, 898, 518, 502, 28, 678, 835, 437, 136, 843, 901, 476, 431, 827, 336, 939]
    
    # 书籍数据集 (SFR-Mistral)
    books_indices = [979, 401, 470, 199, 180, 625, 673, 625, 222, 304, 994, 533, 926, 331, 890, 635, 542, 631, 394, 877, 753, 243, 110, 848, 209, 285, 629, 322, 954, 631, 775, 468, 369, 121, 234, 740, 64, 625, 673, 770, 369, 478, 897, 519, 673, 625, 64, 625, 33, 42, 112, 195, 721, 118, 770, 770, 753, 775, 572, 304, 721, 307, 936, 383, 661, 690, 836, 171, 149, 180, 7, 385, 964, 323, 267, 355, 408, 848, 671, 516, 542, 223, 213, 25, 293, 994, 578, 833, 770, 57, 631, 936, 112, 216, 132, 544, 629, 213, 128, 149, 770, 290, 304, 936, 890, 332, 421, 765, 318, 408, 118, 848, 234, 149, 323, 768, 544, 964, 541, 421, 911, 369, 213, 304, 982, 936, 753, 770, 293, 477, 454, 25, 911, 164, 966, 199, 982, 477, 964, 213, 408, 673, 369, 243, 29, 29, 625, 477, 57, 40, 936, 92, 780, 223, 64, 749, 741, 56, 780, 749, 575, 199, 121, 437, 932, 673, 316, 349, 265, 661, 673, 307, 477, 25, 292, 625, 331, 199, 519, 408, 698, 964, 673, 488, 969, 753, 182, 223, 118, 370, 770, 541, 964, 443, 324, 890, 544, 673, 149, 324]
    
    # 诺贝尔奖数据集 (SFR-Mistral)
    nobel_indices = [565, 378, 904, 407, 32, 152, 931, 222, 682, 881, 816, 265, 378, 70, 273, 206, 54, 217, 940, 143, 973, 925, 904, 465, 840, 848, 960, 245, 929, 416, 616, 321, 258, 237, 788, 55, 874, 301, 797, 509, 736, 54, 645, 881, 365, 2, 651, 141, 903, 290, 873, 231, 81, 258, 837, 428, 286, 152, 527, 273, 717, 523, 379, 97, 206, 54, 143, 614, 608, 217, 331, 651, 231, 973, 639, 237, 222, 407, 689, 832, 285, 19, 998, 781, 652, 881, 645, 984, 421, 682, 990, 70, 527, 472, 570, 378, 235, 365, 875, 837, 781, 133, 416, 91, 881, 124, 144, 365, 383, 705, 722, 407, 284, 875, 17, 206, 133, 348, 431, 365, 845, 503, 688, 111, 144, 974, 2, 688, 875, 81, 538, 206, 655, 837, 538, 39, 991, 515, 81, 111, 217, 245, 545, 90, 786, 374, 572, 471, 565, 845, 19, 875, 19, 787, 998, 269, 881, 2, 206, 471, 840, 508, 788, 143, 0, 682, 904, 457, 998, 509, 321, 302, 503, 70, 570, 608, 383, 39, 527, 940, 574, 126, 81, 70, 32, 273, 152, 903, 508, 416, 998, 707, 837, 990, 385, 118, 407, 400, 137, 538]
    
    # 根据数据集名称返回对应的索引列表
    dataset_map = {
        "basketball_players": basketball_indices,
        "IMDB_name.basics": imdb_indices,
        "books": books_indices,
        "nobel-prize-laureates": nobel_indices
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: {list(dataset_map.keys())}")
    
    return dataset_map[dataset_name]


def get_target_doc_indices_by_model(dataset_name="basketball_players", model_name="SFR-Mistral"):
    """
    根据数据集和模型名称返回目标文档索引列表
    
    Args:
        dataset_name: 数据集名称
        model_name: 模型名称 ("SFR-Mistral", "E5-mistral")
    
    Returns:
        list: 目标文档索引列表
    """
    
    # 篮球运动员 - SFR-Mistral
    basketball_sfr = [721, 762, 145, 947, 765, 132, 397, 137, 791, 68, 594, 290, 448, 183, 591, 768, 137, 657, 762, 955, 637, 859, 873, 572, 92, 939, 640, 450, 863, 439, 448, 884, 223, 951, 976, 892, 563, 769, 631, 146, 506, 841, 653, 628, 146, 587, 835, 433, 863, 148, 859, 953, 976, 259, 860, 146, 110, 719, 458, 935, 130, 686, 719, 912, 139, 390, 768, 858, 937, 739, 534, 552, 727, 637, 458, 62, 995, 92, 976, 891, 413, 703, 873, 976, 769, 290, 884, 148, 290, 840, 490, 146, 873, 469, 65, 467, 525, 762, 891, 939, 450, 990, 727, 492, 625, 422, 884, 390, 211, 137, 211, 492, 884, 381, 458, 768, 663, 806, 772, 873, 592, 976, 835, 492, 835, 148, 886, 891, 824, 718, 313, 763, 829, 450, 995, 645, 706, 924, 995, 525, 390, 572, 390, 139, 976, 525, 492, 458, 679, 891, 703, 629, 249, 976, 132, 381, 703, 467, 492, 841, 452, 249, 313, 840, 92, 492, 525, 433, 886, 597, 718, 141, 259, 672, 504, 719, 840, 703, 513, 146, 886, 504, 58, 631, 62, 727, 62, 62, 552, 60, 769, 726, 858, 939, 838, 806, 223, 506, 637, 911]
    
    # IMDB - SFR-Mistral
    imdb_sfr = [518, 815, 362, 744, 538, 688, 787, 853, 688, 371, 688, 744, 835, 691, 890, 381, 944, 883, 362, 176, 269, 712, 898, 399, 359, 815, 919, 757, 880, 826, 939, 567, 476, 394, 381, 560, 561, 937, 796, 498, 538, 541, 467, 95, 816, 953, 843, 682, 173, 399, 250, 691, 95, 837, 764, 853, 646, 481, 565, 391, 796, 399, 476, 749, 890, 775, 919, 853, 883, 757, 95, 502, 844, 380, 786, 291, 437, 682, 719, 796, 869, 814, 764, 891, 891, 362, 502, 467, 176, 843, 419, 853, 751, 282, 694, 541, 843, 998, 827, 538, 250, 853, 686, 846, 469, 469, 937, 744, 815, 545, 891, 757, 282, 891, 541, 853, 786, 837, 846, 751, 541, 757, 541, 519, 95, 399, 998, 673, 28, 560, 288, 541, 251, 399, 416, 883, 524, 519, 419, 744, 502, 419, 775, 541, 476, 82, 519, 381, 929, 846, 136, 787, 901, 691, 437, 971, 561, 444, 686, 458, 883, 370, 28, 437, 787, 816, 865, 883, 399, 853, 890, 173, 416, 890, 816, 726, 362, 476, 598, 173, 560, 281, 281, 310, 846, 898, 518, 502, 28, 678, 835, 437, 136, 843, 901, 476, 431, 827, 336, 939]
    
    # 书籍 - SFR-Mistral
    books_sfr = [979, 401, 470, 199, 180, 625, 673, 625, 222, 304, 994, 533, 926, 331, 890, 635, 542, 631, 394, 877, 753, 243, 110, 848, 209, 285, 629, 322, 954, 631, 775, 468, 369, 121, 234, 740, 64, 625, 673, 770, 369, 478, 897, 519, 673, 625, 64, 625, 33, 42, 112, 195, 721, 118, 770, 770, 753, 775, 572, 304, 721, 307, 936, 383, 661, 690, 836, 171, 149, 180, 7, 385, 964, 323, 267, 355, 408, 848, 671, 516, 542, 223, 213, 25, 293, 994, 578, 833, 770, 57, 631, 936, 112, 216, 132, 544, 629, 213, 128, 149, 770, 290, 304, 936, 890, 332, 421, 765, 318, 408, 118, 848, 234, 149, 323, 768, 544, 964, 541, 421, 911, 369, 213, 304, 982, 936, 753, 770, 293, 477, 454, 25, 911, 164, 966, 199, 982, 477, 964, 213, 408, 673, 369, 243, 29, 29, 625, 477, 57, 40, 936, 92, 780, 223, 64, 749, 741, 56, 780, 749, 575, 199, 121, 437, 932, 673, 316, 349, 265, 661, 673, 307, 477, 25, 292, 625, 331, 199, 519, 408, 698, 964, 673, 488, 969, 753, 182, 223, 118, 370, 770, 541, 964, 443, 324, 890, 544, 673, 149, 324]
    
    # 诺贝尔 - SFR-Mistral
    nobel_sfr = [565, 378, 904, 407, 32, 152, 931, 222, 682, 881, 816, 265, 378, 70, 273, 206, 54, 217, 940, 143, 973, 925, 904, 465, 840, 848, 960, 245, 929, 416, 616, 321, 258, 237, 788, 55, 874, 301, 797, 509, 736, 54, 645, 881, 365, 2, 651, 141, 903, 290, 873, 231, 81, 258, 837, 428, 286, 152, 527, 273, 717, 523, 379, 97, 206, 54, 143, 614, 608, 217, 331, 651, 231, 973, 639, 237, 222, 407, 689, 832, 285, 19, 998, 781, 652, 881, 645, 984, 421, 682, 990, 70, 527, 472, 570, 378, 235, 365, 875, 837, 781, 133, 416, 91, 881, 124, 144, 365, 383, 705, 722, 407, 284, 875, 17, 206, 133, 348, 431, 365, 845, 503, 688, 111, 144, 974, 2, 688, 875, 81, 538, 206, 655, 837, 538, 39, 991, 515, 81, 111, 217, 245, 545, 90, 786, 374, 572, 471, 565, 845, 19, 875, 19, 787, 998, 269, 881, 2, 206, 471, 840, 508, 788, 143, 0, 682, 904, 457, 998, 509, 321, 302, 503, 70, 570, 608, 383, 39, 527, 940, 574, 126, 81, 70, 32, 273, 152, 903, 508, 416, 998, 707, 837, 990, 385, 118, 407, 400, 137, 538]
    
    # E5-mistral 版本 (如果需要)
    imdb_e5 = [937, 112, 560, 869, 391, 95, 835, 419, 688, 476, 694, 919, 892, 678, 251, 561, 744, 251, 744, 439, 391, 711, 751, 678, 851, 890, 560, 82, 919, 954, 939, 560, 816, 82, 444, 784, 816, 527, 381, 998, 749, 437, 960, 502, 309, 437, 519, 436, 416, 502, 565, 688, 399, 866, 835, 751, 567, 764, 786, 481, 719, 525, 745, 815, 82, 865, 518, 370, 419, 712, 827, 281, 535, 835, 749, 28, 309, 251, 481, 744, 865, 729, 785, 827, 678, 481, 250, 496, 998, 869, 541, 869, 764, 722, 356, 281, 227, 998, 749, 176, 282, 826, 416, 309, 749, 467, 786, 751, 937, 901, 734, 281, 837, 439, 785, 764, 560, 678, 95, 250, 95, 391, 913, 444, 560, 541, 770, 814, 416, 853, 901, 309, 688, 476, 560, 251, 219, 719, 399, 560, 33, 416, 764, 280, 682, 837, 694, 561, 814, 419, 527, 688, 744, 176, 496, 565, 787, 869, 678, 815, 678, 851, 282, 757, 897, 281, 646, 467, 439, 969, 929, 95, 646, 712, 288, 288, 560, 28, 95, 281, 538, 815, 764, 538, 538, 281, 749, 751, 247, 869, 193, 565, 95, 682, 439, 929, 890, 688, 439, 176]
    
    nobel_e5 = [111, 124, 191, 722, 998, 845, 606, 929, 237, 881, 301, 407, 90, 816, 152, 881, 240, 365, 574, 797, 17, 137, 904, 465, 797, 359, 400, 722, 866, 507, 151, 717, 691, 606, 837, 478, 282, 40, 91, 848, 797, 69, 845, 133, 866, 527, 302, 560, 875, 837, 998, 574, 304, 848, 378, 478, 655, 81, 321, 321, 144, 245, 20, 507, 825, 90, 385, 960, 124, 655, 97, 428, 998, 152, 478, 191, 883, 108, 217, 788, 359, 527, 788, 938, 750, 407, 244, 729, 66, 191, 81, 705, 144, 454, 465, 651, 679, 507, 655, 722, 155, 2, 90, 845, 378, 717, 137, 904, 191, 304, 606, 608, 929, 152, 848, 538, 545, 691, 153, 282, 399, 144, 321, 185, 938, 974, 302, 90, 688, 788, 929, 302, 614, 679, 282, 428, 224, 688, 940, 574, 567, 239, 152, 651, 217, 574, 39, 938, 19, 788, 273, 938, 655, 574, 81, 848, 321, 929, 321, 90, 152, 70, 611, 904, 200, 760, 111, 538, 940, 509, 302, 144, 875, 823, 2, 407, 682, 881, 845, 407, 881, 645, 527, 507, 245, 655, 70, 2, 567, 239, 736, 866, 651, 17, 97, 318, 788, 90, 374, 881]
    
    dataset_model_map = {
        ("basketball_players", "SFR-Mistral"): basketball_sfr,
        ("IMDB_name.basics", "SFR-Mistral"): imdb_sfr,
        ("books", "SFR-Mistral"): books_sfr,
        ("nobel-prize-laureates", "SFR-Mistral"): nobel_sfr,
        ("IMDB_name.basics", "E5-mistral"): imdb_e5,
        ("nobel-prize-laureates", "E5-mistral"): nobel_e5,
    }
    
    key = (dataset_name, model_name)
    if key not in dataset_model_map:
        raise ValueError(f"Unknown combination: dataset={dataset_name}, model={model_name}")
    
    return dataset_model_map[key]


