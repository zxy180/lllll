import json
from DGD.help_functions import user_prompt_generation
import torch
from transformers import AutoTokenizer, AutoModel
from DGD.help_functions import last_token_pool
import torch.nn.functional as F
import Levenshtein

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

json_file_path = ""
prompt_text_list = []
adversarial_prompt_list = []
user_querys = []
method = "bba"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
local_model_path = "your_model_address"
tok = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path, torch_dtype=torch.float32).to(device)

with open(json_file_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

if method == "charmer":
    samples_list = json_data.get("samples_list_view", [])
    
    for item in samples_list:
        if "user_query" in item and item["user_query"]:
            user_querys.append(item["user_query"])
        
        if "charmer_adversarial_prompt" in item and item["charmer_adversarial_prompt"]:
            adversarial_prompt_list.append(item["charmer_adversarial_prompt"])

    print("提取完成！")
    print(f"prompt_text 共提取到 {len(prompt_text_list)} 条数据")
    print(f"charmer_adversarial_prompt 共提取到 {len(adversarial_prompt_list)} 条数据")
    
elif method == "bba":
    for key in json_data:
        item = json_data[key]
        if "user_query" in item and item["user_query"]:
            user_querys.append(item["user_query"])
        if "character_query" in item and item["character_query"]:
            adversarial_prompt_list.append(item["character_query"])
            
elif method == "ggpp":    
    for key in json_data:
        item = json_data[key]
        if "user_query" in item and item["user_query"]:
            user_querys.append(item["user_query"])
        if "optimized_query" in item and item["optimized_query"]:
            adversarial_prompt_list.append(item["optimized_query"])
    
elif method == "derag":
    for key in json_data:
        item = json_data[key]
        if "user_query" in item and item["user_query"]:
            user_querys.append(item["user_query"])
        if "opt_query" in item and item["opt_query"]:
            adversarial_prompt_list.append(item["opt_query"])
            
elif method == "gcg":
    for key in json_data:
        item = json_data[key]
        if "user_query" in item and item["user_query"]:
            user_querys.append(item["user_query"])
        if "optimized_query" in item and item["optimized_query"]:
            adversarial_prompt_list.append(item["optimized_query"])

edit_distances = []
sim_list = []

for sent1, sent2 in zip(user_querys, adversarial_prompt_list):
    edit_distance = Levenshtein.distance(sent1, sent2)
    edit_distances.append(edit_distance)
    
    similarity_ratio = calculate_query_similarity(sent1, sent2, model, tok, device)
    sim_list.append(similarity_ratio)
    
    edit_distance_weighted = Levenshtein.editops(sent1, sent2)
    
    print(f"原始句子与篡改句子的【编辑距离】: {edit_distance}")
    print(f"原始句子与篡改句子的【相似度】: {similarity_ratio:.4f}")
    print(f"具体的编辑操作明细: {edit_distance_weighted}")
    
avg_edit_distance = sum(edit_distances) / len(user_querys)
avg_similarity_ratio = sum(sim_list) / len(user_querys)

print(f"总的编辑距离：{sum(edit_distances)}")
print(f"总的相似度{sum(sim_list)}")
print(f"平均编辑距离：{avg_edit_distance:.2f}") 
print(f"平均相似度：{avg_similarity_ratio:.2f}")
