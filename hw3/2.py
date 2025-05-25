from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys

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
    train_documents = [item['answer'] for item in train_data]
    train_questions = [item['query'] for item in train_data]
    test_documents = [item['answer'] for item in test_data]
    test_questions = [item['query'] for item in test_data]
except KeyError as e:
    print(f"Ошибка: Ожидаемое поле {e} не найдено в датасете.")
    print("Пожалуйста, проверьте доступные колонки:")
    print(dataset.column_names)
    raise

test_pairs = list(zip(test_questions, test_documents))
print("Настройка TF-IDF векторизатора на обучающих документах...")
tfidf_vectorizer = TfidfVectorizer()
if train_documents:
    tfidf_vectorizer.fit(train_documents)
    print("TF-IDF векторизатор настроен.")
else:
    print("Ошибка: Обучающая выборка документов пуста. Невозможно настроить векторизатор.")
    raise ValueError("Обучающая выборка документов пуста.")

print("Векторизация вопросов и документов тестовой выборки...")
if test_questions and test_documents:
    test_question_vectors = tfidf_vectorizer.transform(test_questions)
    test_document_vectors = tfidf_vectorizer.transform(test_documents)
    print("Векторизация завершена.")
else:
    print("Ошибка: Тестовая выборка вопросов или документов пуста. Невозможно векторизовать.")
    raise ValueError("Тестовая выборка вопросов или документов пуста.")

print("Расчет косинусной близости и ранжирование...")
if test_question_vectors.shape[0] > 0 and test_document_vectors.shape[0] > 0:
    similarity_matrix = cosine_similarity(test_question_vectors, test_document_vectors)
    print("Расчет близости завершен.")
else:
    print("Ошибка: Векторы тестовой выборки пусты. Невозможно рассчитать близость.")
    raise ValueError("Векторы тестовой выборки пусты.")

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


    print("\n--- Результаты TF-IDF Baseline ---")
    print(f"Количество тестовых вопросов: {num_test_questions}")
    print(f"MRR: {mean_mrr:.4f}")
    print(f"Recall@1: {final_recall_at_1:.4f}")
    print(f"Recall@3: {final_recall_at_3:.4f}")
    print(f"Recall@10: {final_recall_at_10:.4f}")
