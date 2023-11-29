import joblib
from src.features.text_conversion import text_conversion

text = input('Введите описание: ')

clear_text = text_conversion(text)

TFIDF_model = joblib.load('models/TFIDF_model.joblib')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

X_new = tfidf_vectorizer.transform([clear_text])
y_pred_new = TFIDF_model.predict_proba(X_new)
print(f'Вероятность успеха (больше 1 000 загрузок в неделю): {y_pred_new[0][1] * 100:.2f}%')