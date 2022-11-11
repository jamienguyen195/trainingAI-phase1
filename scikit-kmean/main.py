import numpy as np


def get_tf_idf(data_path, output_path):
    data_tf_idf = []
    with open('../TF-IDF/20news-bydate/word_idf.txt') as f:
        word_idf = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        idf = dict(word_idf)

    with open(data_path) as f:
        docs = f.read().splitlines()

    for doc in docs[:2000]:
        doc = doc.split('<fff>')
        label, name, content = int(doc[0]), int(doc[1]), doc[2]
        words = [word for word in content.split() if word in idf]
        unique_word = list(set(words))
        max_tf = max([words.count(word) for word in unique_word])
        word_tfidf = []
        for word, idf_val in word_idf:
            tf_idf = words.count(word) * 1. / max_tf * idf_val
            word_tfidf.append(str(tf_idf))
        dense_vec = ' '.join(word_tfidf)
        data_tf_idf.append((label, name, dense_vec))
    with open(output_path, 'w') as f:
        f.write(
            '\n'.join(str(label) + '<fff>' + str(name) + '<fff>' + dense_vec for label, name, dense_vec in data_tf_idf))


get_tf_idf('../TF-IDF/20news-bydate/news_data_train', '20news-train-tfidf.txt')
get_tf_idf('../TF-IDF/20news-bydate/news_data_test', '20news-test-tfidf.txt')


def load_data(data_path):
    with open(data_path) as f:
        lines = f.read().splitlines()
    labels = [int(line.split('<fff>')[0]) for line in lines]
    data = [[float(value) for value in line.split('<fff>')[2].split()] for line in lines]
    return data, labels



def clustering_with_KMeans():
    data, label = load_data('20news-train-tfidf.txt')
    from sklearn.cluster import KMeans
    print('=========')
    kmeans = KMeans(n_clusters=20,
                    init='random',
                    n_init=5,  # number of running times with different centroids
                    tol=1e-3,  # threshold
                    random_state=1).fit(data)
    labels = kmeans.labels_


def classifying_with_linear_SVMs():
    trainX, trainY = load_data('20news-train-tfidf.txt')
    print(trainY)
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(C=10.,
                           tol=1e-3,
                           verbose=True)
    classifier.fit(trainX, trainY)
    testX, testY = load_data('20news-test-tfidf.txt')
    predictedY = classifier.predict(testX)
    accuracy = compute_accuracy(predictedY, testY)
    print('accuracy: ', accuracy)


def compute_accuracy(predictedY, ground_truth):
    matches = np.equal(predictedY, ground_truth)
    accuracy = np.sum(matches.astype(float))/predictedY.size
    return accuracy


classifying_with_linear_SVMs()
