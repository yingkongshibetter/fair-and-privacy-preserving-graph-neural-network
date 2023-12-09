import torch
import torch.nn.functional as F
import torch.nn as nn

def simi(output):  # new_version
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)
    return res
  
def avg_err(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    the_maxs, _ = torch.max(x_corresponding, 1)
    the_maxs = the_maxs.reshape(the_maxs.shape[0], 1).repeat(1, x_corresponding.shape[1])
    c = 2 * torch.ones_like(x_corresponding)
    x_corresponding = ( c.pow(x_corresponding) - 1) / c.pow(the_maxs)
    the_ones = torch.ones_like(x_corresponding)
    new_x_corresponding = torch.cat((the_ones, 1 - x_corresponding), 1)
    for i in range(x_corresponding.shape[1] - 1):
        x_corresponding = torch.mul(x_corresponding, new_x_corresponding[:, -x_corresponding.shape[1] - 1 - i : -1 - i])
    the_range = torch.arange(0., x_corresponding.shape[1]).repeat(x_corresponding.shape[0], 1) + 1
    score_rank = (1 / the_range[:, 0:]) * x_corresponding[:, 0:]
    final = torch.mean(torch.sum(score_rank, axis=1))
    print("Now Average ERR@k = ", final.item())
    return final.item()
  
def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    print("Now Average NDCG@k = ", avg_ndcg.item())
    return avg_ndcg.item()

def lambdas_computation_only_review(x_similarity, y_similarity, top_k):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    length_of_k = k_para * top_k - 1
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)
    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]
    return x_sorted_scores, y_sorted_idxs, x_corresponding

def NDCG (output, features):
  y_similarity = simi(output])
  x_similarity = simi(features)
  x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity,y_similarity, top_k)
  return avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k))

def ERR (output, features):
  y_similarity = simi(output])
  x_similarity = simi(features)
  x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity,y_similarity, top_k)
  return avg_err(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def privacy (embeding,epoch,sens,idx_sensitive,idx_nosensitive)
  adv_inferds = nn.Linear(args.hidden, 2)
  optimizer = optim.Adam(adv_inferds.parameters(), lr=0.01)
  for i in range(epoch):
    adv_inferds.train()
    optimizer.zero_grad()
    s_g = adv_inferds(embeding.detach())[idx_sensitive]
    a_loss = F.cross_entropy(s_g,sens[idx_sensitive])
    a_loss.backward()
    optimizer.step()
  adv_inferds.eval()
  y_pred = adv_inferds(embeding)
  privacy_acc = accuracy(y_pred, sens[idx_nosensitive])
  return privacy_acc
  
