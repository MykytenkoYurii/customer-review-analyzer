import json
from src.preprocessing import preprocess_text
from src.sentiment_analysis import analyze_sentiment
from src.keyword_extraction import extract_keywords

BATCH_SIZE = 100

with open("data/sample_reviews.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

data = []

#Обробляємо тексти
for line in lines[:BATCH_SIZE]:
    if line.startswith("__label__"):
        label = int(line[9])
        text = line[11:].strip()
        cleaned = preprocess_text(text)
        sentiment = analyze_sentiment(cleaned)
        data.append({
            "label": label,
            "text": text,
            "cleaned": cleaned,
            "sentiment": sentiment
        })

#Після того, як зібрано data — обчислюємо ключові слова
corpus = [d["cleaned"] for d in data]
keywords_list = extract_keywords(corpus)

for i in range(len(data)):
    data[i]["keywords"] = keywords_list[i]

with open("data/processed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✔ Обробка завершена. Дані збережено у data/processed.json")
