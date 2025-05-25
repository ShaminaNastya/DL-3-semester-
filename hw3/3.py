from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

print("Загрузка датасета sentence-transformers/natural-questions...")
dataset = load_dataset("sentence-transformers/natural-questions", split='train')
print("Датасет загружен.")
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

except KeyError as e:
    print(f"Ошибка: Ожидаемое поле {e} не найдено в датасете.")
    print("Пожалуйста, проверьте доступные колонки:")
    print(dataset.column_names)
    raise ValueError(f"Отсутствует необходимое поле в датасете: {e}")

if not test_questions or not test_documents:
     print("Ошибка: Тестовая выборка вопросов или документов пуста.")
     raise ValueError("Тестовая выборка пуста.")

model_name = "intfloat/multilingual-e5-base"
print(f"Загрузка модели {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
print("Модель загружена.")

def get_embeddings(texts, model, tokenizer, device, prefix=""):
    processed_texts = [prefix + t for t in texts]
    encoded_input = tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = average_pool(model_output.last_hidden_state, encoded_input['attention_mask'])

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[:, None]

print("Векторизация тестовых вопросов и документов с помощью E5...")
batch_size = 64
test_question_embeddings = []
for i in range(0, len(test_questions), batch_size):
    batch_questions = test_questions[i:i+batch_size]
    embeddings = get_embeddings(batch_questions, model, tokenizer, device, prefix="query: ")
    test_question_embeddings.append(embeddings)
test_question_embeddings = np.concatenate(test_question_embeddings, axis=0)

test_document_embeddings = []
for i in range(0, len(test_documents), batch_size):
    batch_documents = test_documents[i:i+batch_size]
    embeddings = get_embeddings(batch_documents, model, tokenizer, device, prefix="passage: ")
    test_document_embeddings.append(embeddings)
test_document_embeddings = np.concatenate(test_document_embeddings, axis=0)

print("Векторизация завершена.")
print(f"Форма эмбеддингов вопросов: {test_question_embeddings.shape}")
print(f"Форма эмбеддингов документов: {test_document_embeddings.shape}")

print("Расчет косинусной близости...")
similarity_matrix = cosine_similarity(test_question_embeddings, test_document_embeddings)
print("Расчет близости завершен.")

print("Расчет метрик MRR и Recall@k...")
mrr_scores = []
recall_at_1 = 0
recall_at_3 = 0
recall_at_10 = 0

num_test_questions = len(test_questions)
if num_test_questions == 0:
    print("Ошибка: Количество тестовых вопросов равно нулю. Невозможно рассчитать метрики.")
else:
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
    final_recall_at_1 = recall_at_1 / num_test_questions if num_test_questions > 0 else 0
    final_recall_at_3 = recall_at_3 / num_test_questions if num_test_questions > 0 else 0
    final_recall_at_10 = recall_at_10 / num_test_questions if num_test_questions > 0 else 0

    print("\n--- Результаты E5 Baseline ---")
    print(f"Количество тестовых вопросов: {num_test_questions}")
    print(f"MRR: {mean_mrr:.4f}")
    print(f"Recall@1: {final_recall_at_1:.4f}")
    print(f"Recall@3: {final_recall_at_3:.4f}")
    print(f"Recall@10: {final_recall_at_10:.4f}")
