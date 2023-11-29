import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from translatepy import Translator
from langdetect import detect

def translate_to_english(text):
    """
    Анализ языка, с автоматическим переводом на английский, если он отличается
    """
    detect_language = detect(text)
    translator = Translator()
    if detect_language != 'en':
        result = translator.translate(text, 'en')
        return result.result
    else:
        return text
    
def preprocess_text(text):
    """
    Приведение текста к общему виду
    """
    # загружаем список стоп-слов для английского языка
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))

    # текст к нижнему регистру
    text = text.lower()
    # удаление неалфавитных символов и цифр
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    # токенизация
    words = text.split()
    # удаление стоп-слов
    words = [word for word in words if word not in stop_words]
    # лемматизация
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)

    return text

def text_conversion(text):
    """
    Использует вышеописанные функции для приведения
    описания мобильного приложения к общему виду
    """
    # применяем функцию к копии основного датасета
    eng_text = translate_to_english(text)
    # применяем функцию предобработки
    clear_text = preprocess_text(eng_text)

    return clear_text