import torch
import torch.nn as nn
import torch.nn.functional as F

class ExactMatchFocalLoss(nn.Module):
    def __init__(self, weights_list=None, alpha=1.0, gamma=2.0):
        super(ExactMatchFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights_list = weights_list # Danh sách trọng số cho 6 đầu ra

    def forward(self, outputs, targets):
        total_loss = 0
        individual_correct = []

        for i in range(6):
            w = self.weights_list[i] if self.weights_list is not None else None
            
            # THÊM label_smoothing=0.1
            # Nó sẽ biến nhãn [0, 1, 0] thành [0.033, 0.933, 0.033]
            ce_loss = F.cross_entropy(
                outputs[i], 
                targets[:, i], 
                reduction='none', 
                weight=w,
                label_smoothing=0.1  # Giá trị vàng trong training
            )
            
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            total_loss += focal_loss.mean()
            
            pred = outputs[i].argmax(dim=1)
            individual_correct.append(pred == targets[:, i])

        all_correct = torch.stack(individual_correct, dim=1).all(dim=1)
        em_penalty = (1.0 - all_correct.float()).mean() * self.alpha
        
        return total_loss + em_penalty