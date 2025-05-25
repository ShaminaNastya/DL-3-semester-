from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

os.environ["WANDB_DISABLED"] = "true"
print("Weights & Biases отключен.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

print("Загрузка датасета sentence-transformers/natural-questions...")
try:
    dataset = load_dataset("sentence-transformers/natural-questions", split='train')
    print("Датасет загружен.")
except ValueError as e:
    print(f"Ошибка при загрузке датасета: {e}")
    print("Возможно, требуется обновить библиотеки datasets и fsspec: pip install --upgrade datasets fsspec")
    raise

print(f"Загружен сплит с {len(dataset)} записями.")
print("Доступные колонки:", dataset.column_names)

print("Разделение датасета на train (80%) и test (20%)...")
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset['train']
test_data = split_dataset['test']
print("Датасет разделен.")
print(f"Размер обучающей выборки: {len(train_data)}")
print(f"Размер тестовой выборки: {len(test_data)}")

try:
    i = train_data[0]['query']
    j = train_data[0]['answer']

    test_documents = [item['answer'] for item in test_data]
    test_questions = [item['query'] for item in test_data]
    train_documents = [item['answer'] for item in train_data]
    train_questions = [item['query'] for item in train_data]


except KeyError as e:
    print(f"Ошибка: Ожидаемое поле {e} не найдено в датасете.")
    print("Пожалуйста, проверьте доступные колонки:")
    print(dataset.column_names)
    raise ValueError(f"Отсутствует необходимое поле в датасете: {e}")

if not test_questions or not test_documents or not train_questions or not train_documents:
     print("Ошибка: Одна из выборок (train/test вопросы/документы) пуста.")
     raise ValueError("Выборка пуста.")

train_size_limit = 5000
print(f"\nПодготовка данных для дообучения с Hard Negatives (используется {train_size_limit} примеров)...")

train_examples_hard_negatives = []
for i in range(min(train_size_limit, len(train_data))):
    item = train_data[i]
    anchor = item['query']
    positive = item['answer']
    train_examples_hard_negatives.append(InputExample(texts=[anchor, positive], label=i))

print(f"Подготовлено {len(train_examples_hard_negatives)} примеров для дообучения с Hard Negatives.")

model_name = "intfloat/multilingual-e5-base"
num_epochs = 1
train_batch_size = 64

def custom_pairwise_cos_sim(embeddings):
    return util.cos_sim(embeddings, embeddings)


print(f"\nДообучение модели с BatchHardTripletLoss (эпох: {num_epochs}, батч: {train_batch_size}, примеров: {len(train_examples_hard_negatives)})...")
model_hard_negatives = SentenceTransformer(model_name, device=device)
train_dataloader_hard_negatives = DataLoader(train_examples_hard_negatives, shuffle=True, batch_size=train_batch_size)
train_loss_hard_negatives = losses.BatchHardTripletLoss(model=model_hard_negatives, distance_metric=custom_pairwise_cos_sim, margin=0.5)

warmup_steps_hard_negatives = int(len(train_dataloader_hard_negatives) * num_epochs * 0.1) if len(train_dataloader_hard_negatives) > 0 else 0
output_path_hard_negatives = "output/e5-hard-negatives-finetuned"
model_hard_negatives.fit(train_objectives=[(train_dataloader_hard_negatives, train_loss_hard_negatives)],
                      epochs=num_epochs,
                      warmup_steps=warmup_steps_hard_negatives,
                      output_path=output_path_hard_negatives)
print("Дообучение с BatchHardTripletLoss завершено.")

def evaluate_model(model, test_questions, test_documents):
    print(f"\nОценка модели...")
    print("Векторизация тестовых вопросов и документов...")
    test_question_embeddings = model.encode(test_questions, convert_to_numpy=True, show_progress_bar=False)
    test_document_embeddings = model.encode(test_documents, convert_to_numpy=True, show_progress_bar=False)
    print("Векторизация завершена.")

    print(f"Форма эмбеддингов вопросов: {test_question_embeddings.shape}")
    print(f"Форма эмбеддингов документов: {test_document_embeddings.shape}")

    print("Расчет косинусной близости...")
    if test_question_embeddings.shape[0] == 0 or test_document_embeddings.shape[0] == 0:
        print("Ошибка: Пустые эмбеддинги тестовой выборки.")
        return {"MRR": 0, "Recall@1": 0, "Recall@3": 0, "Recall@10": 0}

    similarity_matrix = util.cos_sim(test_question_embeddings, test_document_embeddings).numpy()
    print("Расчет близости завершен.")

    print("Расчет метрик MRR и Recall@k...")
    mrr_scores = []
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_10 = 0
    num_test_questions = len(test_questions)

    if num_test_questions == 0:
        print("Ошибка: Количество тестовых вопросов равно нулю.")
        return {"MRR": 0, "Recall@1": 0, "Recall@3": 0, "Recall@10": 0}

    for i in range(num_test_questions):
        true_document_index = i
        question_similarity_scores = similarity_matrix[i]
        ranked_document_indices = np.argsort(question_similarity_scores)[::-1]
        rank = -1
        indices_of_true_doc = np.where(ranked_document_indices == true_document_index)[0]
        if indices_of_true_doc.size > 0:
            rank = indices_of_true_doc[0] + 1
            mrr_scores.append(1.0 / rank)
            if rank <= 1:
                recall_at_1 += 1
            if rank <= 3:
                recall_at_3 += 1
            if rank <= 10:
                recall_at_10 += 1

    mean_mrr = np.mean(mrr_scores) if mrr_scores else 0
    final_recall_at_1 = recall_at_1 / num_test_questions
    final_recall_at_3 = recall_at_3 / num_test_questions
    final_recall_at_10 = recall_at_10 / num_test_questions
    results = {
        "MRR": mean_mrr,
        "Recall@1": final_recall_at_1,
        "Recall@3": final_recall_at_3,
        "Recall@10": final_recall_at_10
    }
    return results

print("\n--- Результаты E5 BatchHardTripletLoss Fine-tuned ---")
try:
    model_hard_negatives_loaded = SentenceTransformer(output_path_hard_negatives, device=device)
    results_hard_negatives = evaluate_model(model_hard_negatives_loaded, test_questions, test_documents)
    print(results_hard_negatives)
except Exception as e:
    print(f"Ошибка при оценке модели с BatchHardTripletLoss: {e}")
    results_hard_negatives = {"MRR": 0, "Recall@1": 0, "Recall@3": 0, "Recall@10": 0}

print(f"E5 Triplet (Hard Negatives) Fine-tuned: {results_hard_negatives}")
