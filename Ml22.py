from preprocessing import TextPreprocessor
from analytics import ChatAnalytics
from gensim import corpora, models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

prep = TextPreprocessor()
stats = ChatAnalytics(prep)

stats.load_data('data/halz_tima.txt')

print("\n" + "="*30)
print(" СТАТИСТИКА ПО АВТОРАМ")
print("="*30)
print(stats.get_top_users().to_string())

print("\n" + "="*30)
print(" АКТИВНОСТЬ ПО ЧАСАМ")
print("="*30)
print(stats.get_activity_by_hour().to_string())

print("\n" + "="*30)
print(" ТОП 10 ПОПУЛЯРНЫХ СЛОВ")
print("="*30)
print(stats.get_common_words(10).to_string())

stats.plot_stats()

lda_documents = []

for text in stats.df['text']:
	clean_tokens = prep.clean_text(text, strict_filter=True)

	if clean_tokens:
		lda_documents.append(clean_tokens)

print(f"Подготовлено {len(lda_documents)} сообщений для анализа.")

dictionary = corpora.Dictionary(lda_documents)
corpus = [dictionary.doc2bow(doc) for doc in lda_documents]

lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=25)

for idx, topic in lda_model.print_topics(-1):
    print(f"Тема {idx + 1}: {topic}")


