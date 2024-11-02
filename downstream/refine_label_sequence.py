import os
import re
import pickle
import numpy as np
from pathlib import Path
from parameters import parse_args
from sklearn.metrics import accuracy_score, precision_score, f1_score, precision_recall_fscore_support
from data.DA_20K.one_level_label_dict import label2metalabel, label_key_index_dict


def refine_label_sequence(True_labels, Retrieval_labels_with_scores, Reranking_metalabels_with_scores):
    all_data = 0
    Rerank_hit = 0
    Retrieval_hit = 0
    Retrieval_precision_at_top_k = 0
    Retrieval_precision_at_top_k_rank = 0
    Retrieval_recall_at_top_1 = 0
    Retrieval_recall_at_top_k = 0

    k = 2
    y_true_all = []
    y_pred_all = []
    y_rerank_all = []
    for ques_id, label_truth_cur in True_labels.items():
        all_data += 1
        y_true = [0 for _ in range(len(label_key_index_dict))]
        for one_true in label_truth_cur:
            y_true[int(label_key_index_dict[one_true])] = 1
            y_true_all.append(y_true)

            metalabels_with_scores_cur = Reranking_metalabels_with_scores[ques_id]
            metalabels_with_scores_cur = {one.split('\t')[0]: float(one.split('\t')[1]) for one in
                                          metalabels_with_scores_cur}

            Retrieval_labels_with_scores_cur = Retrieval_labels_with_scores[ques_id]
            Retrieval_labels_sequence = [one.split('\t')[0] for one in Retrieval_labels_with_scores_cur]
            Retrieval_labels_with_scores_cur = {one.split('\t')[0]: float(one.split('\t')[1]) for one in
                                                Retrieval_labels_with_scores_cur}
            Label_Pred_top1 = max(Retrieval_labels_with_scores_cur, key=Retrieval_labels_with_scores_cur.get)

            y_pred = [0 for _ in range(len(label_key_index_dict))]
            for one_pred in [Label_Pred_top1]:
                y_pred[int(label_key_index_dict[one_pred])] = 1
            y_pred_all.append(y_pred)

            one_hit = [0.0 for i in range(len(Retrieval_labels_sequence))]
            for i in range(len(Retrieval_labels_sequence)):
                one_retrieval_label = Retrieval_labels_sequence[i]
                metalabels_cur = label2metalabel[one_retrieval_label]
                for metalabel in metalabels_cur:
                    if metalabel in metalabels_with_scores_cur.keys():
                        one_hit[i] += metalabels_with_scores_cur[metalabel]
                one_hit[i] /= len(metalabels_cur)
            one_hit = [one_hit[index] * Retrieval_labels_with_scores_cur[key] for index, key in
                       enumerate(Retrieval_labels_sequence)]
            Refine_labels_with_scores_cur = sorted(list(zip(Retrieval_labels_sequence, one_hit)), key=lambda x: x[1],
                                                   reverse=True)
            Refine_labels_sequence = [item[0] for item in Refine_labels_with_scores_cur]

            y_rerank = [0 for _ in range(len(label_key_index_dict))]
            y_rerank[int(label_key_index_dict[Refine_labels_sequence[0]])] = 1
            y_rerank_all.append(y_rerank)

            if Label_Pred_top1 in label_truth_cur:
                Retrieval_hit += 1

            Refine_top1_label_idx = one_hit.index(max(one_hit))  # 如果预测了多个最大值，只保留最前面的那个
            Refine_top1_label = Retrieval_labels_sequence[Refine_top1_label_idx]
            if Refine_top1_label in label_truth_cur:
                Rerank_hit += 1

            precision_at_top_k = 0
            for Retrieval_label in Retrieval_labels_sequence[:k]:
                if Retrieval_label in label_truth_cur:
                    precision_at_top_k += 1
            precision_at_top_k /= k
            Retrieval_precision_at_top_k += precision_at_top_k

            precision_at_top_k_rerank = 0
            for Rank_Bert_Pred in Refine_labels_sequence[:k]:
                if Rank_Bert_Pred in label_truth_cur:
                    precision_at_top_k_rerank += 1
            precision_at_top_k_rerank /= k
            Retrieval_precision_at_top_k_rank += precision_at_top_k_rerank

    P_1 = round(Retrieval_hit / all_data, 4)
    R_1 = round(Retrieval_recall_at_top_1 / all_data, 4)
    P_1_rerank = round(Rerank_hit / all_data, 4)
    P_k = round(Retrieval_precision_at_top_k / all_data, 4)
    R_k = round(Retrieval_recall_at_top_k / all_data, 4)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_rerank_all = np.array(y_rerank_all)
    print('Question Numbers:' + str(all_data))
    print('Retrieval P@1:' + str(P_1))
    print('Retrieval MicroF1@1:' + str(round(f1_score(y_true_all, y_pred_all, average='micro', zero_division=1), 4)))
    print('Retrieval MacroF1@1:' + str(round(f1_score(y_true_all, y_pred_all, average='macro', zero_division=1), 4)))
    print('Reranking P@1:' + str(P_1_rerank))
    print('Reranking MicroF1@1:' + str(round(f1_score(y_true_all, y_rerank_all, average='micro', zero_division=1), 4)))
    print('Reranking MacroF1@1:' + str(round(f1_score(y_true_all, y_rerank_all, average='macro', zero_division=1), 4)))
    print()


if __name__ == '__main__':
    args = parse_args()
    out_path = Path(args.out_path) / f'{args.name}-{args.dataset}'
    out_model_path = out_path / 'Reranking_trained_model'
    out_result_path = out_path / 'Reranking_result'
    out_label_path = out_path / 'label'
    out_metalabel_path = out_path / 'metalabel'

    true_label_path = out_label_path / 'True_labels.pkl'
    with open(true_label_path, 'rb') as f:
        True_labels = pickle.load(f)
    Retrieval_labels_with_scores_path = out_label_path / 'Retrieval_labels_with_scores'
    with open(Retrieval_labels_with_scores_path, 'rb') as f:
        Retrieval_labels_with_scores = pickle.load(f)

    for metalabel_sequence in os.listdir(out_metalabel_path):
        path = out_metalabel_path / metalabel_sequence
        Ver = re.findall(r'Reranking_metalabels_with_scores_loop(.*)', metalabel_sequence)[0]
        print('Ver:{}'.format(int(Ver)))
        with open(path, 'rb') as f:
            Reranking_metalabels_with_scores = pickle.load(f)
        refine_label_sequence(True_labels=True_labels, Retrieval_labels_with_scores=Retrieval_labels_with_scores,
                              Reranking_metalabels_with_scores=Reranking_metalabels_with_scores)
