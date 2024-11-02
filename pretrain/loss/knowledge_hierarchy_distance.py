import pandas as pd
from tqdm import tqdm
from data.Math_Junior.four_level_label_dict import label_key_index_dict


class KHDistance(object):
    def __init__(self, know_path, data_path):
        self.know_df = pd.read_csv(know_path)
        self.label_paths = self.precompute_paths()
        self.lines = self.read_lines(data_path)
        self.label_group_distance_cache = {}
        self.rank_list = self.process_knowledge_in_batches()

    @staticmethod
    def read_lines(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                parts = line.strip().split('\t')
                label_indices = []
                for i in range(len(parts[2:])):
                    if (i + 1) % 4 == 0:
                        idx = str(label_key_index_dict[parts[i + 2]])
                        label_indices.append(idx)
                lines.append('\t'.join(parts[:2]) + '\t' + '\t'.join(label_indices))
        return lines

    def precompute_paths(self):
        label_paths = {}
        for _, row in self.know_df.iterrows():
            path = [row['label1'], row['label2'], row['label3'], row['label4']]
            if path:
                label_paths[path[-1]] = path
        return label_paths

    def compute_distance(self, i, j):
        path_i = self.label_paths.get(i, [])
        path_j = self.label_paths.get(j, [])

        if not path_i or not path_j:
            return len(self.know_df.columns)

        common_ancestor_length = 0
        max_common_length = min(len(path_i), len(path_j))

        for k in range(max_common_length):
            if path_i[k] == path_j[k]:
                common_ancestor_length += 1
            else:
                break

        distance_ij = (len(path_i) - common_ancestor_length) + (len(path_j) - common_ancestor_length)
        return distance_ij

    def process_knowledge_in_batches(self, batch_size=1000):
        KHDistance = set()
        num_lines = len(self.lines)
        all = 0

        for start in tqdm(range(0, num_lines, batch_size), total=(num_lines + batch_size - 1) // batch_size,
                          desc="Processing Batches"):
            batch = self.lines[start:start + batch_size]
            for i, target in enumerate(batch):
                all += 1
                target_label_group_indices = tuple(sorted(map(int, target.split('\t')[2:])))

                for j, ques in enumerate(batch):
                    if j <= i:
                        continue
                    label_group_indices = tuple(sorted(map(int, ques.split('\t')[2:])))

                    # 创建标签组对的键，保证 (group1, group2) 和 (group2, group1) 一致
                    label_pair = (target_label_group_indices,
                                  label_group_indices) if target_label_group_indices < label_group_indices else (
                        label_group_indices, target_label_group_indices)

                    # 检查标签组之间的距离是否已经缓存
                    if label_pair not in self.label_group_distance_cache:
                        KHDistance_cur = 0
                        flag = False
                        for t_idx in target_label_group_indices:
                            for k_idx in label_group_indices:
                                distance = self.compute_distance(t_idx, k_idx)
                                if distance == 0:
                                    flag = True
                                    break
                                KHDistance_cur += distance
                            if flag:
                                break
                        # avg_KHDistance = round(KHDistance_cur / (len(target_label_group_indices) * len(label_group_indices)), 4)
                        if flag:
                            # avg_KHDistance = 0
                            KHDistance_cur = 0
                        # 存储距离到缓存
                        # self.label_group_distance_cache[label_pair] = avg_KHDistance
                        self.label_group_distance_cache[label_pair] = KHDistance_cur
                    else:
                        avg_KHDistance = self.label_group_distance_cache[label_pair]

                    # KHDistance.add(avg_KHDistance)
                    KHDistance.add(KHDistance_cur)

        print(all)

        # 跨批次的距离处理
        for start in tqdm(range(0, num_lines, batch_size), total=(num_lines + batch_size - 1) // batch_size,
                          desc="Handling Cross-Batch Distances"):
            batch = self.lines[start:start + batch_size]
            for i, target in enumerate(batch):
                target_label_group_indices = tuple(sorted(map(int, target.split('\t')[2:])))

                for prev_start in range(0, start, batch_size):
                    prev_batch = self.lines[prev_start:prev_start + batch_size]
                    for j, ques in enumerate(prev_batch):
                        label_group_indices = tuple(sorted(map(int, ques.split('\t')[2:])))  # 将标签组排序并转换为元组

                        label_pair = (target_label_group_indices,
                                      label_group_indices) if target_label_group_indices < label_group_indices else (
                            label_group_indices, target_label_group_indices)

                        if label_pair not in self.label_group_distance_cache:
                            KHDistance_cur = 0
                            flag = False
                            for t_idx in target_label_group_indices:
                                for k_idx in label_group_indices:
                                    distance = self.compute_distance(t_idx, k_idx)
                                    if distance == 0:
                                        flag = True
                                        break
                                    KHDistance_cur += distance
                                if flag:
                                    break
                            # avg_KHDistance = round(KHDistance_cur / (len(target_label_group_indices) * len(label_group_indices)), 4)
                            if flag:
                                # avg_KHDistance = 0
                                KHDistance_cur = 0
                            # 缓存跨批次的距离
                            # self.label_group_distance_cache[label_pair] = avg_KHDistance
                            self.label_group_distance_cache[label_pair] = KHDistance_cur
                        else:
                            # avg_KHDistance = self.label_group_distance_cache[label_pair]
                            KHDistance_cur = self.label_group_distance_cache[label_pair]

                        # KHDistance.add(avg_KHDistance)
                        KHDistance.add(KHDistance_cur)

        print(all)
        return sorted(KHDistance)

# data_path = f'data/our_data/train2.csv'
# know_path = f'data/our_data/label_tree_index.csv'
# criterion = KHDistance(know_path, data_path, args=None) # 16min

# import pickle

# with open('ranking_contrastive_loss.pkl', 'wb') as f:
#     pickle.dump(criterion, f)

# with open('ranking_contrastive_loss.pkl', 'rb') as f:
#     criterion = pickle.load(f)
# print(len(criterion.rank_list))  # 135 / 96(不平均)
# lines = criterion.lines
# num_lines = len(lines)
