import torch

def Class_Centering_Learning_loss(logits, labels, class_center):
    batch_size = logits.shape[0]
    loss_CCL_all = torch.zeros(1, dtype=torch.float, device=labels.device)
    for i in range(batch_size):
        label_indices = torch.where(labels[i] == 1)[0].cpu()
        loss_CCL = torch.sum(torch.cdist(logits[i].unsqueeze(0), class_center[label_indices])) / len(label_indices)
        loss_CCL_all += loss_CCL
    loss_CCL_all /= batch_size
    return loss_CCL_all
