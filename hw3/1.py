def recall_at_k(target: list[int], predict: list[list[int]], k: int) -> float:

    if not target or not predict or len(target) != len(predict):
        raise ValueError("target and predict must be non-empty lists of the same length.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    num_queries = len(target)
    recalls = 0
    for i in range(num_queries):
        correct_doc_id = target[i]
        predicted_docs_at_k = predict[i][:k]

        if correct_doc_id in predicted_docs_at_k:
            recalls += 1

    return recalls / num_queries if num_queries > 0 else 0.0

def mean_reciprocal_rank(target: list[int], predict: list[list[int]]) -> float:

    if not target or not predict or len(target) != len(predict):
        raise ValueError("target and predict must be non-empty lists of the same length.")

    num_queries = len(target)
    reciprocal_ranks = 0.0
    for i in range(num_queries):
        correct_doc_id = target[i]
        predicted_docs = predict[i]
        try:
            rank = predicted_docs.index(correct_doc_id) + 1
            reciprocal_ranks += 1.0 / rank
        except ValueError:
            reciprocal_ranks += 0.0

    return reciprocal_ranks / num_queries if num_queries > 0 else 0.0

target_ids = [1, 5, 2]
predicted_ids = [[10, 20, 1, 30, 40],
     [50, 5, 60, 70],
     [80, 90, 100]]
k = 3
recall_k = recall_at_k(target_ids, predicted_ids, k)
print(f"Recall@{k}: {recall_k}")
mrr_score = mean_reciprocal_rank(target_ids, predicted_ids)
print(f"MRR: {mrr_score}")
