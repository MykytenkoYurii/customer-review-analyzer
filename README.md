
A Python tool for analyzing customer reviews: extracting keywords and detecting sentiment using NLP.

Проєкт для аналізу відгуків клієнтів з використанням Python та NLP.

## 🔍 Що робить цей скрипт:

- Витягує **текст відгуків** з `.txt` файлу
- Очищає текст (препроцесінг)
- Визначає **тональність** (позитив/негатив)
- Витягує **ключові слова** (TF-IDF)
- Зберігає результат у `processed.json`
- Візуалізує дані в `notebook/eda.ipynb`

## 🧪 Приклад структури обробленого запису:

```json
{
  "label": 2,
  "text": "Great CD! My lovely Pat has one of the GREAT voices...",
  "cleaned": "great cd my lovely pat has one of the great voices",
  "sentiment": "positive",
  "keywords": ["cd", "just", "mood"]
}

