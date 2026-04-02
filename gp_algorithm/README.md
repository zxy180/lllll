**Environment Setup**
Run the following command to install dependencies:
pip install -r requirements.txt
**Running Steps**
Step 1: Generate Retrieval Results
Run the main DBO attack script to generate adversarial queries and retrieve results from the RAG system
python gp_algorithm.py
Output: Retrieval result JSON files
Function: Implements the discrete Bayesian optimization with Gaussian Process (GP) surrogate model, searches the optimal character perturbation, and outputs the retrieval results of adversarial queries.
Step 2: Calculate Edit Distance & Similarity
Process the generated retrieval JSON files to compute perturbation metrics.
python get_modify.py
Input: Retrieval JSON files from Step 1
Output: Updated JSON files with edit distance (Levenshtein distance) and semantic similarity between original/perturbed queries
Step 3: Evaluate Generate Attack Success Rate
Run the inference script to calculate the final attack performance.
bash
python infer.py




