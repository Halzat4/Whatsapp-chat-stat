import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pymorphy3
from collections import Counter

nltk.download('stopwords', quiet=True)
morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))
en_stemmer = SnowballStemmer('english')


def preprocess_chat(text):
	text = re.sub(r'\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2} - .*сквозным шифрованием.*\n?', '', text)

	text = text.replace("Тима Казну", "Тимур")
	text = text.replace("Халзат Турсунов", "Халзат")

	lines = text.split('\n')
	processed_lines = []

	word_freq = Counter()
	user_stats = Counter()

	for line in lines:
		line = line.strip()
		if not line:
			continue

		match = re.match(r'^\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2} - (.*?): (.*)$', line)

		if match:
			sender = match.group(1)
			content = match.group(2)
			user_stats[sender] += 1
		else:
			if re.match(r'^\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2} - ', line):
				continue
			sender = ""
			content = line

		if "<Без медиафайлов>" in content:
			message_body = "МЕДИА"
		elif "Данное сообщение удалено" in content:
			message_body = "УДАЛЕНО"
		else:
			content = content.lower()
			content = re.sub(r'[^а-яА-Яa-zA-Z\s]', ' ', content)
			tokens = content.split()

			cleaned_tokens = []
			for word in tokens:
				if re.search('[а-яА-Я]', word):
					word = morph.parse(word)[0].normal_form
				elif re.search('[a-zA-Z]', word):
					word = en_stemmer.stem(word)

				if word not in stop_words and len(word) > 1:
					cleaned_tokens.append(word)

			word_freq.update(cleaned_tokens)
			message_body = " ".join(cleaned_tokens)

		if sender:
			final_line = f"{sender.lower()} {message_body}".strip()
		else:
			final_line = message_body.strip()

		if final_line:
			processed_lines.append(final_line)

	return processed_lines, word_freq, user_stats

file_path = 'data/halz_tima.txt'

with open(file_path, 'r', encoding='utf-8') as f:
	chat_content = f.read()

result, word_counts, sender_counts = preprocess_chat(chat_content)

print(f"Обработка завершена. Всего строк: {len(result)}")
print("-" * 30)

print("СТАТИСТИКА ПО ОТПРАВИТЕЛЯМ:")
for user, count in sender_counts.items():
	print(f"{user}: {count} сообщений")

print("\nТОП 10 СЛОВ:")
for word, freq in word_counts.most_common(10):
	print(f"{word}: {freq}")

print("-" * 30)
print("ПЕРВЫЕ 10 СТРОК ЧАТА:")
for line in result[:10]:
	print(line)
