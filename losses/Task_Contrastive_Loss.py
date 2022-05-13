import torch
import torch.nn.functional as F
import random

class TaskContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self):
        super(TaskContrastiveLoss, self).__init__()

    def shuffle_unrepeat(self, order):
        if len(order) <= 1:
            raise ValueError('Len of shuffle should be greater than 1')
        record_order = [i for i in order]
        shuffle_unrepeat_order = []
        count = 0
        last_value = None
        while len(record_order) > 0:
            value = random.choice(record_order)
            if value != order[count]:
                shuffle_unrepeat_order.append(value)
                record_order.remove(value)
                count += 1
            else:
                if (count + 1) == len(order):
                    if value == order[count]:
                        last_value = value
                        shuffle_unrepeat_order.append(value)
                        record_order.remove(value)
                    break
        if last_value:
            random_idx = random.randint(0, len(shuffle_unrepeat_order)-2)
            shuffle_unrepeat_order[-1] = shuffle_unrepeat_order[random_idx]
            shuffle_unrepeat_order[random_idx] = last_value
        
        return shuffle_unrepeat_order

    def rand_task_features(self, feature):
        task_size = feature.shape[0]
        task_id = [i for i in range(task_size)]
        task_id_shuffled = self.shuffle_unrepeat(task_id)
        feature_shuffle_unrepeat = feature[task_id_shuffled]
        return feature_shuffle_unrepeat

    def forward(self, output1, output2):
        inner_class_distance = F.pairwise_distance(output1, output2, keepdim=True)
        output2_shuffle_unrepeat = self.rand_task_features(output2)
        cross_class_distance = F.pairwise_distance(output1, output2_shuffle_unrepeat, keepdim=True)
        
        inner_class_MSE = torch.pow(inner_class_distance, 2)
        cross_class_MSE = torch.pow(cross_class_distance, 2)
        loss_contrastive = torch.mean(torch.log(1 + inner_class_MSE.exp_()) + torch.log(1 + (-cross_class_MSE).exp_()))
        return loss_contrastive, torch.mean(inner_class_distance), torch.mean(cross_class_distance)

if __name__ == "__main__":
    output1 = torch.rand(8, 64, 1, 1)
    output2 = torch.rand(8, 64, 1, 1)
    loss = TaskContrastiveLoss()
    counter = 0
    while 1:
        print(loss(output1, output2))
        counter += 1
        print(counter)