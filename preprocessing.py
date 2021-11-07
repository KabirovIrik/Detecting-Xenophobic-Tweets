import pandas as pd
import numpy as np
import spacy
import string
import re
import nltk
from collections import Counter
from spellchecker import SpellChecker
from defines import *


class PrepareText:
    def __init__(self, df, to_lower=True, remove_emoji=True, 
        remove_emoticons=True, chat_words_convert=True,
        spell_correct=True, add_space=True, remove_punct=True,
        remove_stopwords=True, remove_freq=False, remove_rares=False,
        remove_url=True, get_lemma=False):

        self.TO_LOWER = to_lower
        self.REMOVE_EMOJI = remove_emoji
        self.REMOVE_EMOTICONS = remove_emoticons
        self.CHAT_WORDS_CONVERSION = chat_words_convert
        self.SPELL_CORRECT = spell_correct
        self.ADD_SPACE = add_space
        self.REMOVE_PUNCT = remove_punct
        self.REMOVE_STOPWORDS = remove_stopwords
        self.REMOVE_FREQ = remove_freq
        self.REMOVE_RARES = remove_rares
        self.REMOVE_URL = remove_url
        self.GET_LEMMA = get_lemma

        self.df = df

        # Create our list of punctuation marks
        self.punctuations = string.punctuation

        # Create our list of stopwords
        self.nlp = spacy.load('en_core_web_lg')
        self.STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS

    def save_file(self, df_name='prepared_df.csv'):
        self.df.to_csv(df_name, sep=';', index=False)

    def prepare(self):
        # string to lower
        print('preparing')
        if self.TO_LOWER:
            print('TO_LOWER')
            self.df["Text"] = self.df["Text"].str.lower()

        # Removal of Emojis
        # Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
        if self.REMOVE_EMOJI:
            print('REMOVE_EMOJI')
            self.df["Text"] = self.df["Text"].apply(PrepareText.remove_emoji)

        if self.REMOVE_EMOTICONS:
            print('REMOVE_EMOTICONS')
            self.df["Text"] = self.df["Text"].apply(PrepareText.remove_emoticons)

        if self.CHAT_WORDS_CONVERSION:
            print('CHAT_WORDS_CONVERSION')
            chat_words_list, chat_words_map_dict = PrepareText.get_chat_words()
            self.df["Text"] = self.df["Text"].apply(PrepareText.chat_words_conversion, args=(chat_words_list, chat_words_map_dict))

        if self.SPELL_CORRECT:
            print('SPELL_CORRECT')
            # Spelling Correction
            spell = SpellChecker()
            self.df["Text"] = self.df["Text"].apply(PrepareText.correct_spellings, args=(spell,))

        # add spaces after dots and comma
        if self.ADD_SPACE:
            print('ADD_SPACE')
            self.df["Text"] = self.df["Text"].apply(PrepareText.addSpace)

        # remove punct
        if self.REMOVE_PUNCT:
            print('REMOVE_PUNCT')
            PUNCT_TO_REMOVE = string.punctuation
            self.df["Text"] = self.df["Text"].apply(PrepareText.remove_punctuation, args=(PUNCT_TO_REMOVE,))

        if self.REMOVE_STOPWORDS:
            print('REMOVE_STOPWORDS')
            self.df["Text"] = self.df["Text"].apply(PrepareText.remove_stopwords, args=(self.STOPWORDS,))

        if self.REMOVE_FREQ:
            print('REMOVE_FREQ')
            # Freq words
            cnt = Counter()
            for text in self.df["Text"].values:
                for word in text.split():
                    cnt[word] += 1
                    
            # Removal of Frequent words
            FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])

            self.df["Text"] = self.df["Text"].apply(PrepareText.remove_freqwords, args=(FREQWORDS,))

        # Removal of Rare words
        if self.REMOVE_RARES:
            print('REMOVE_RARES')
            n_rare_words = 10
            RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
            self.df["Text"] = self.df["Text"].apply(PrepareText.remove_rarewords, args=(RAREWORDS,))

        if self.REMOVE_URL:
            print('REMOVE_URL')
            # Removal of URL
            self.df["Text"] = self.df["Text"].apply(PrepareText.remove_urls)

        if self.GET_LEMMA:
            print('GET_LEMMA')
            # Lemmatization
            self.df["Text"] = self.df["Text"].apply(PrepareText.get_lemma, args=(self.nlp,))

    @staticmethod
    def get_chat_words():
        # Chat Words Conversion
        chat_words_map_dict = {}
        chat_words_list = []
        for line in chat_words_str.split("\n"):
            if line != "":
                cw = line.split("=")[0]
                cw_expanded = line.split("=")[1]
                chat_words_list.append(cw)
                chat_words_map_dict[cw] = cw_expanded
        return set(chat_words_list), chat_words_map_dict

    @staticmethod
    def remove_emoji(string):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    @staticmethod
    def remove_emoticons(text):
        emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
        return emoticon_pattern.sub(r'', text)


    @staticmethod
    def chat_words_conversion(text, chat_words_list, chat_words_map_dict):
        new_text = []
        for w in text.split():
            if w.upper() in chat_words_list:
                new_text.append(chat_words_map_dict[w.upper()])
            else:
                new_text.append(w)
        return " ".join(new_text)

    @staticmethod
    def correct_spellings(text, spell):
        corrected_text = []
        misspelled_words = spell.unknown(text.split())
        for word in text.split():
            if word in misspelled_words:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)
        return " ".join(corrected_text)

    @staticmethod
    def addSpace(val):
        return re.sub(r'(\.+|\,+)', r'\1 ', val)

    @staticmethod
    def remove_punctuation(text, PUNCT_TO_REMOVE):
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

    @staticmethod
    def remove_stopwords(text, STOPWORDS):
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

    @staticmethod
    def remove_freqwords(text, FREQWORDS):
        return " ".join([word for word in str(text).split() if word not in FREQWORDS])

    @staticmethod
    def remove_rarewords(text, RAREWORDS):
        return " ".join([word for word in str(text).split() if word not in RAREWORDS])

    @staticmethod
    def remove_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    @staticmethod
    def get_lemma(text, nlp):
        doc = nlp(text)
        return ' '.join([tok.lemma_ for tok in doc])

#df = pd.read_csv('TrainingDS.csv')
#df_sample = pd.read_csv('Sample Submission.csv')
#df_test = pd.read_csv('TestingDS.csv')
