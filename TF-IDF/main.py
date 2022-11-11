import os
from nltk.stem.porter import PorterStemmer
import regex as re
from collections import defaultdict
import numpy as np


def gather_20newsgroup_data():
    path = '20news-bydate/'
    dir_name = [path + dir + '/' for dir in os.listdir(path) if not os.path.isfile(path + dir)]
    train_dir, test_dir = (dir_name[0], dir_name[1]) if "train" in dir_name[0] else (dir_name[1], dir_name[0])
    list_news_group = [news_group for news_group in os.listdir(train_dir) if
                       os.path.isdir(train_dir + news_group + '/')]
    list_news_group.sort()
    return list_news_group


list_news_group = gather_20newsgroup_data()
# print(list_news_group)

with open('20news-bydate/stopwords.txt') as f:
    stop_words = f.read().splitlines()


def collect_data_from_newsgroup(parent_dir, newsgroup_list):
    stemmer = PorterStemmer()
    data = []
    for group_id, newsgroup in enumerate(newsgroup_list):
        label = group_id
        dir_path = parent_dir + '/' + newsgroup + '/'
        files = [(files_name, dir_path + files_name) for files_name in os.listdir(dir_path) if
                 os.path.isfile(dir_path + files_name)]
        files.sort()
        for file_name, file_path in files:
            try:
                with open(file_path) as f:
                    text = f.read().lower()
                    words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
                    content = " ".join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + file_name + '<fff>' + content)
            except:
                pass

    return data


train_data = collect_data_from_newsgroup('20news-bydate/20news-bydate-train', list_news_group)
with open('20news-bydate/news_data_train', 'w') as f:
    f.write('\n'.join(line for line in train_data))
test_data = collect_data_from_newsgroup('20news-bydate/20news-bydate-test', list_news_group)
with open('20news-bydate/news_data_test', 'w') as f:
    f.write('\n'.join(line for line in test_data))
full_data = train_data + test_data
with open('20news-bydate/news_data', 'w') as f:
    f.write('\n'.join(line for line in full_data))


def compute_idf(df, corpus_size):
    assert df > 0
    return np.log10(corpus_size / df)


def generate_vocab(data_path):
    with open(data_path) as f:
        docs = f.read().splitlines()
    doc_count = defaultdict(int)
    num_doc = len(docs)

    for doc in docs:
        content = doc.split('<fff>')
        text = content[-1]
        words = list(set(text.split()))
        for word in set(words):
            doc_count[word] += 1

    words_idf = [(word, compute_idf(df, num_doc))
                 for word, df in zip(doc_count.keys(), doc_count.values())
                 if not word.isdigit()]

    sorted(words_idf, key=lambda idf: idf[-1])
    print("vocab size: ", len(words_idf))
    with open('20news-bydate/word_idf.txt', 'w') as f:
        f.write('\n'.join(word + '<fff>' + str(idf) for word, idf in words_idf))


generate_vocab('20news-bydate/news_data')


def get_tf_idf(data_path, file_to_write):
    data_tf_idf = []
    with open('20news-bydate/word_idf.txt') as f:
        word_idf = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        word_ID = dict([(word, index) for index, (word, idf) in enumerate(word_idf)])
        idf = dict(word_idf)

    with open(data_path) as f:
        docs = f.read().splitlines()

    for doc in docs:
        doc = doc.split('<fff>')
        label, name, content = int(doc[0]), int(doc[1]), doc[2]
        words = [word for word in content.split() if word in idf]
        unique_word = list(set(words))
        max_tf = max([words.count(word) for word in unique_word])
        word_tfidf = []
        sum_square = 0
        for word in unique_word:
            term_count = words.count(word)
            tf_idf_value = term_count / max_tf * idf[word]
            word_tfidf.append((word_ID[word], tf_idf_value))
            sum_square += tf_idf_value ** 2
        word_tfidf_normalized = [str(index) + ':' + str(tf_idf_value / sum_square ** 0.5)
                                 for (index, tf_idf_value) in word_tfidf]
        sparse_vec = ' '.join(word_tfidf_normalized)
        data_tf_idf.append((label, name, sparse_vec))
    with open(file_to_write, 'w') as f:
        f.write(
            '\n'.join(str(label) + '<fff>' + str(name) + '<fff>' + sparse_vec for label, name, sparse_vec in data_tf_idf))


get_tf_idf('20news-bydate/news_data', '20news-bydate/data_tf_idf.txt')
get_tf_idf('20news-bydate/news_data_test', '20news-bydate/data_tf_idf_test.txt')
get_tf_idf('20news-bydate/news_data_train', '20news-bydate/data_tf_idf_train.txt')