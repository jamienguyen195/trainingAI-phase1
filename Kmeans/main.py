from collections import defaultdict
import numpy as np

PATH = '../TF-IDF/20news-bydate/data_tf_idf.txt'


class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self.r_d = r_d
        self.label = label
        self.doc_id = doc_id


class Cluster:
    def __init__(self):
        self.centroid = None
        self.members = []

    def reset_member(self):
        self.members = []

    def add_member(self, member):
        self.members.append(member)


class Kmeans:
    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.clusters = [Cluster() for _ in range(num_cluster)]
        self.E = []  # list of centroids
        self.S = 0  # overall similarity

    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            r_d = [0. for _ in range(vocab_size)]
            indices_tfidf = sparse_r_d.split()
            for index_tfidf in indices_tfidf:
                index = int(index_tfidf.split(':')[0])
                tf_idf = float(index_tfidf.split(':')[1])
                r_d[index] = tf_idf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open('../TF-IDF/20news-bydate/word_idf.txt') as f:
            vocab_size = len(f.read().splitlines())

        self.data = []
        self.label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            content = d.split('<fff>')
            label, doc_id = int(content[0]), int(content[1])
            self.label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=content[2], vocab_size=vocab_size)
            self.data.append(Member(r_d=r_d, label=label, doc_id=doc_id))

    def random_init(self, seed_val):
        np.random.seed(seed_val)
        choices = np.random.choice(len(self.data), self.num_cluster, replace=False)  # 20: number of clusters
        for index, choice in enumerate(choices):
            self.E.append(self.data[choice].r_d)
            self.clusters[index].centroid = self.data[choice].r_d

    def run(self, seed_val, criterion, threshold):
        self.random_init(seed_val)
        self.iteration = 0
        while True:
            for cluster in self.clusters:
                cluster.reset_member()

            self.new_S = 0
            for member in self.data:
                max_s = self.select_cluster_for(member)
                self.new_S += max_s

            for cluster in self.clusters:
                self.update_centroid_of(cluster)

            self.iteration += 1
            if self.stopping_condition(criterion, threshold):
                break

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self.clusters:
            similarity = self.compute_similarity(member, cluster.centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity

    def compute_similarity(self, member, centroid):
        # print('member', member)
        # print('centroid', centroid)
        return member.r_d.dot(centroid)

    def update_centroid_of(self, cluster):
        member_r_ds = [member.r_d for member in cluster.members]
        avg_r_d = np.mean(member_r_ds, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(avg_r_d ** 2))
        new_centroid = avg_r_d / sqrt_sum_sqr
        cluster.centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            if self.iteration >= threshold:
                return True
            else:
                return False
        elif criterion == 'centroid':
            E_new = [list(cluster.centroid) for cluster in self.clusters]
            E_new_except_E = [centroid for centroid in E_new if centroid not in self.E]
            self.E = E_new
            if len(E_new_except_E) <= threshold:
                return True
            else:
                return False
        else:
            delta_S = self.new_S - self.S
            self.S = self.new_S
            if delta_S <= threshold:
                return True
            else:
                return False

    def compute_purity(self):
        majority_sum = 0
        for cluster in self.clusters:
            member_labels = [member.label for member in cluster.members]
            max_count = max([member_labels.count(label) for label in range(self.num_cluster)])
            majority_sum += max_count
        return majority_sum * 1. / len(self.data)  # len(self.data) = number of doc

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self.data)
        for cluster in self.clusters:
            wk = len(cluster.members) * 1.
            H_omega += wk / N * np.log10(wk / N)
            member_labels = [member.label for member in cluster.members]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self.label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)

        for label in range(20):
            cj = self.label_count[label] * 1.
            H_C = -cj / N * np.log10(cj / N)
        return I_value * 2. / (H_omega + H_C)


# k_mean = Kmeans(20)
# k_mean.load_data(PATH)
# k_mean.run(0, 'max_iters', 10 ** 3)
# print(k_mean.compute_purity())
