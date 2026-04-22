import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class ChatAnalytics:
	def __init__(self, preprocessor):
		self.preprocessor = preprocessor
		self.data = []
		self.df = None

	def load_data(self, file_path):
		self.data = []
		name_map = {
			'Халзат Турсунов': 'Халзат',
			'Тима Казну': 'Тимур'
		}

		with open(file_path, 'r', encoding='utf-8') as f:
			for line in f:
				parsed = self.preprocessor.parse_line(line)
				if parsed:
					parsed['author'] = name_map.get(parsed['author'], parsed['author'])
					parsed['tokens'] = self.preprocessor.clean_text(parsed['text'], strict_filter=False)
					self.data.append(parsed)

		self.df = pd.DataFrame(self.data)
		print(f"Успешно загружено {len(self.df)} сообщений.")

	def get_top_users(self):
		if self.df is None or self.df.empty: return "Нет данных"
		return self.df['author'].value_counts().rename_axis('Автор').rename('Количество')

	def get_activity_by_hour(self):
		if self.df is None or self.df.empty: return "Нет данных"
		if 'hour' not in self.df.columns:
			self.df['hour'] = self.df['time'].str.split(':').str[0]
		return self.df['hour'].value_counts().sort_index().rename_axis('Часы').rename('Сообщения')

	def get_common_words(self, top_n=10):
		if self.df is None or self.df.empty: return pd.Series()

		all_words = [word for tokens in self.df['tokens'] for word in tokens]
		word_counts = Counter(all_words).most_common(top_n)

		words, counts = zip(*word_counts) if word_counts else ([], [])
		return pd.Series(counts, index=words).rename_axis('Слово').rename('Частота')

	def plot_stats(self):
		if self.df is None or self.df.empty: return

		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

		hourly_data = self.get_activity_by_hour()
		hourly_data.plot(kind='line', marker='o', ax=ax1, color='#2ecc71')
		ax1.set_title('Активность по часам')
		ax1.grid(True, alpha=0.3)

		user_data = self.get_top_users()
		user_data.plot(kind='bar', ax=ax2, color=['#3498db', '#e74c3c'])
		ax2.set_title('Кто больше писал')
		plt.setp(ax2.get_xticklabels(), rotation=0)

		word_data = self.get_common_words(10)
		word_data.plot(kind='barh', ax=ax3, color='#f1c40f')
		ax3.set_title('Топ 10 слов')
		ax3.invert_yaxis()

		plt.tight_layout()
		plt.show()