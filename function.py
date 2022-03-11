from tkinter import N
import torch
import torch.nn.functional as f
import numpy as np

from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import DBSCAN, KMeans
from hdbscan import HDBSCAN

from einops import rearrange
import gc


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean_std_2dim(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 2)
    
    if size[0] == 1:
        feat_std = torch.zeros(size[1])
    else:
        feat_var = feat.var(dim=0) + eps
        feat_std = feat_var.sqrt()

    feat_mean = feat.mean(dim=0)

    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


# new method: content-based clustering and cluster-wise AdaIN

# x의 channel을 idx와 나머지로 split한다.
def split_channel(x, idx):
    num_channels = x.shape[1]

    rest = torch.ones(num_channels)
    rest[idx] = 0
    rest_idx = torch.arange(num_channels)[rest == 1]

    return x[:, idx], x[:, rest_idx]


# x1과 x1의 channel index, x2와 x2의 channel index를 주면 x1과 x2를 channel-wise merge한다.
def merge_channel(x1, idx1, x2, idx2):
    merged = torch.cat((x1, x2), dim=1)

    idx = torch.cat((idx1, idx2))
    _, true_idx = torch.topk(idx, len(idx), largest=False)
    
    return merged[:, true_idx]


# n개의 index(0 ~ n-1) 중에서 주어진 idx를 제외한 나머지 index를 return한다.
def find_rest(total, idx): 
    rest = torch.ones(total)
    rest[idx] = 0
    rest_idx = torch.arange(total)[rest == 1]

    return rest_idx

def cluster_AdaIN(content_feats, style_feats):
    num_channels, content_size = content_feats.shape[1], content_feats.shape[2:]
    style_size = style_feats.shape[2:]

    content_points = content_feats.reshape(num_channels, content_size[0] * content_size[1]).T
    style_points = style_feats.reshape(num_channels, style_size[0] * style_size[1]).T

    # density-based clustering
    # dbscan_content = DBSCAN(eps=2.5, min_samples=8)
    # dbscan_style = DBSCAN(eps=2.5, min_samples=8)

    # content_cluster_ids = torch.tensor(dbscan_content.fit_predict(content_primary_with_pos), dtype=torch.int32)
    # style_cluster_ids = torch.tensor(dbscan_style.fit_predict(style_primary_with_pos), dtype=torch.int32)
    
    hdbscan_content = HDBSCAN(min_cluster_size=64, min_samples=1)
    hdbscan_style = HDBSCAN(min_cluster_size=64, min_samples=1)

    content_cluster_ids = torch.tensor(hdbscan_content.fit_predict(content_points), dtype=torch.int32)

    # 각각의 cluster에 대해 mean(cluster center) & std 계산
    num_content_clusters = content_cluster_ids.unique().shape[0]
    if content_cluster_ids.unique()[0] == -1:
        num_content_clusters -= 1

    content_cluster_mean = []
    content_cluster_std = []
    for k in range(num_content_clusters):
        content_idx = torch.nonzero(content_cluster_ids == k).squeeze()
        content_points_in_cluster = torch.index_select(content_points, 0, content_idx)

        mean_content, std_content = calc_mean_std_2dim(content_points_in_cluster)

        content_cluster_mean.append(mean_content)
        content_cluster_std.append(std_content)

    print("# content clusters:", num_content_clusters)

    mean_style, std_style = calc_mean_std_2dim(style_points)

    # assign noise points to clusters
    noise_index = torch.nonzero(content_cluster_ids == -1).squeeze()
    num_noise = noise_index.shape[0]
    if num_noise:
        noise = torch.index_select(content_points, 0, noise_index)
        print(f"total points: {content_size[0] * content_size[1]}")
        print(f"noise: {num_noise}\n")

        clustered_index = torch.nonzero(content_cluster_ids != -1).squeeze()
        clustered_points = torch.index_select(content_points, 0, clustered_index)

        # noise_cluster_ids = kmeans_predict(noise_primary, content_cluster_centers)
        # content_cluster_ids[noise_index] = noise_cluster_ids

        noise_cluster_ids = []
        for noise_point in noise:
            dist = torch.norm(clustered_points - noise_point, dim=1)
            _, knn_idx = dist.topk(max(64, num_content_clusters + 1), largest=False)
            # (content_size[0] * content_size[1] - num_noise) // 16
            knn_clusters = content_cluster_ids[clustered_index[knn_idx]]
            noise_cluster_ids.append(torch.mode(knn_clusters, 0).values)

        content_cluster_ids[noise_index] = torch.tensor(noise_cluster_ids)

    for i in range(content_size[0] * content_size[1]):
        cluster = content_cluster_ids[i]

        normalized = (content_points[i] - content_cluster_mean[cluster]) / (content_cluster_std[cluster] + 1e-7)
        content_points[i] = std_style * normalized + mean_style

    result = content_points.T.reshape(content_feats.shape)
    return result


def kmeans_AdaIN(content_feats, style_feats, k_clusters=16):
    num_channels, content_size = content_feats.shape[1], content_feats.shape[2:]
    style_size = style_feats.shape[2:]

    content_feats_ori = content_feats.clone()

    content_points = content_feats.reshape(num_channels, content_size[0] * content_size[1]).T
    style_points = style_feats.reshape(num_channels, style_size[0] * style_size[1]).T

    # k-means clustering
    content_cluster_ids, content_cluster_centers = kmeans(content_points, num_clusters=k_clusters)

    # 각각의 cluster에 대해 mean(cluster center) & std 계산
    content_cluster_mean = []
    content_cluster_std = []
    for k in range(k_clusters):
        content_idx = torch.nonzero(content_cluster_ids == k).squeeze()
        content_points_in_cluster = torch.index_select(content_points, 0, content_idx)

        mean_content, std_content = calc_mean_std_2dim(content_points_in_cluster)

        content_cluster_mean.append(mean_content)
        content_cluster_std.append(std_content)

    # mean_content, std_content = calc_mean_std_2dim(content_points)
    mean_style, std_style = calc_mean_std_2dim(style_points)

    # AdaIN
    for i in range(content_size[0] * content_size[1]):
        cluster = content_cluster_ids[i]

        normalized = (content_points[i] - content_cluster_mean[cluster]) / (content_cluster_std[cluster] + 1e-7)
        # normalized = (content_points[i] - mean_content) / std_content
        content_points[i] = std_style * normalized + mean_style
        
    result = content_points.T.reshape(content_feats.shape)
    result = 1 * result + 0 * content_feats_ori
    return result


def kmeans_AdaIN_old(content_feats, style_feats, k_clusters=16):
    num_channels, content_size = content_feats.shape[1], content_feats.shape[2:]
    style_size = style_feats.shape[2:]

    content_feats_ori = content_feats.clone()
    content_points = content_feats.reshape(num_channels, content_size[0] * content_size[1]).T
    style_points = style_feats.reshape(num_channels, style_size[0] * style_size[1]).T

    # channel-wise normalization (instance normalization)
    content_points_mean, content_points_std = calc_mean_std_2dim(content_points)
    style_points_mean, style_points_std = calc_mean_std_2dim(style_points)

    content_points_normalized = (content_points - content_points_mean) / (content_points_std + 1e-5)
    style_points_normalized = (style_points - style_points_mean) / (style_points_std + 1e-5)

    # k-means clustering
    content_cluster_ids, content_cluster_centers = kmeans(content_points, num_clusters=k_clusters)
    style_cluster_ids, style_cluster_centers = kmeans(style_points, num_clusters=k_clusters)

    # 각각의 cluster에 대해 mean(cluster center) & std 계산
    ## content
    content_cluster_mean = []
    content_cluster_mean_normalized = []
    content_cluster_std = []
    for k in range(k_clusters):
        content_idx = torch.nonzero(content_cluster_ids == k).squeeze()
        content_points_in_cluster = torch.index_select(content_points, 0, content_idx)

        mean_content, std_content = calc_mean_std_2dim(content_points_in_cluster)

        content_cluster_mean.append(mean_content)
        content_cluster_std.append(std_content)

        content_points_in_cluster_normalized = torch.index_select(content_points_normalized, 0, content_idx)
        content_cluster_mean_normalized.append(calc_mean_std_2dim(content_points_in_cluster_normalized)[0])

    ## style
    style_cluster_mean = []
    style_cluster_mean_normalized = []
    style_cluster_std = []
    for k in range(k_clusters):
        style_idx = torch.nonzero(style_cluster_ids == k).squeeze()
        style_points_in_cluster = torch.index_select(style_points, 0, style_idx)

        mean_style, std_style = calc_mean_std_2dim(style_points_in_cluster)

        style_cluster_mean.append(mean_style)
        style_cluster_std.append(std_style)

        style_points_in_cluster_normalized = torch.index_select(style_points_normalized, 0, style_idx)
        style_cluster_mean_normalized.append(calc_mean_std_2dim(style_points_in_cluster_normalized)[0])

    # cluster matching
    cluster_matching = kmeans_predict(torch.stack(content_cluster_mean_normalized), torch.stack(style_cluster_mean_normalized))

    # AdaIN
    for i in range(content_size[0] * content_size[1]):
        content_cluster = content_cluster_ids[i]
        style_cluster = cluster_matching[content_cluster]

        normalized = (content_points[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-7)
        content_points[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]
        # content_points[i] = content_points[i] - content_cluster_mean[content_cluster] + style_cluster_mean[style_cluster]

    result = content_points.T.reshape(content_feats.shape)
    result = 1 * result + 0 * content_feats_ori
    return result


def cal_p(self, cf, sf):
    """
    :param cf: [c*kernel_size, hcwc]
    :param sf: [c*kernel_size, hsws]
    :param mask: [hcwc, hsws]
    :return:p [c*c]
    """
    cf_size = cf.size()
    sf_size = sf.size()
    
    k_cross = 5

    cf_temp = cf.clone()
    sf_temp = sf.clone()
    # pdb.set_trace()

    # ########################################
    # normalize
    cf_n = f.normalize(cf, 2, 0)
    sf_n = f.normalize(sf, 2, 0)
    # #########################################

    dist = torch.mm(cf_n.t(), sf_n)  # inner product,the larger the value, the more similar

    hcwc, hsws = cf_size[1], sf_size[1]
    U = torch.zeros(hcwc, hsws).type_as(cf_n).cuda()  # construct affinity matrix "(h*w)*(h*w)"

    index = torch.topk(dist, k_cross, 0)[1]  # find indices k nearest neighbors along row dimension
    value = torch.ones(k_cross, hsws).type_as(cf_n).cuda() # "KCross*(h*w)"
    U.scatter_(0, index, value)  # set weight matrix
    del index
    del value
    gc.collect()

    index = torch.topk(dist, k_cross, 1)[1]  # find indices k nearest neighbors along col dimension
    value = torch.ones(hcwc, k_cross).type_as(cf_n).cuda()
    U.scatter_(1, index, value)  # set weight matrix
    del index
    del value
    gc.collect()

    n_cs = torch.sum(U)
    U = U / n_cs

    # WITH ORTHOGONAL CONSTRAINT
    A = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
    # regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.device) * 1e-12
    # A += regularization_term
    A_U, A_S, A_V = torch.svd(A)
    p = torch.mm(A_U, A_V.t())

    return p

    
def cal_p_attn(self, cf, sf):
    """
    :param cf: [c*kernel_size, hcwc]
    :param sf: [c*kernel_size, hsws]
    :param mask: [hcwc, hsws]
    :return:p [c*c]
    """

    cf_temp = cf.clone()
    sf_temp = sf.clone()

    # ########################################
    # normalize
    cf_n = f.normalize(cf, 2, 0)
    sf_n = f.normalize(sf, 2, 0)
    # #########################################

    # normalized cosine similarity
    sim = torch.mm(cf_n.t(), sf_n)  # inner product,the larger the value, the more similar / cosine similarity
    U = sim / torch.sum(sim)

    # # euclidean distance
    # dist = torch.cdist(cf_n.T, sf_n.T, p=2) # euclidean distance
    # sim = 1 / dist
    # U = sim / torch.sum(sim)


    # WITH ORTHOGONAL CONSTRAINT
    A = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
    # regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.device) * 1e-12
    # A += regularization_term
    A_U, A_S, A_V = torch.svd(A)
    p = torch.mm(A_U, A_V.t())

    return p


def mast(self, x, x_detach, y_inter):
    """
    :param ori_cf:
    :param cf: [n, c*kernel_size, hcwc]
    :param sf: [n, c*kernel_size, hsws]
    :param mask: [hcwc, hsws]
    :return: csf [n, c*kernel_size, hcwc]
    """
    # x_m, x_v, y_m, y_v = x_m.detach(), x_v.detach(), y_m.detach(), y_v.detach()
    # x = (((x - x_m) / (x_v+1e-5).sqrt()) * (y_v+1e-5).sqrt()) + y_m
    n_cluster = 3
    x_size = x.size()
    y_inter_size = y_inter.size()
    cf, sf = self.down_sampling_feature(x_detach, y_inter)
    cf_size = cf.size()
    sf_size = sf.size()

    hc, wc = cf_size[2], cf_size[3]
    hs, ws = sf_size[2], sf_size[3]

    x = rearrange(x, 'b c h w -> c (b h w)')
    cf = rearrange(cf, 'b c h w -> c (b h w)')
    sf = rearrange(sf, 'b c h w -> c (b h w)')

    p = self.cal_p(cf, sf)
    # p = self.cal_p_attn(cf, sf)

    csf = torch.mm(p.t(), x)

    # pdb.set_trace()
    csf = rearrange(csf, 'c (b h w) -> b c h w', b=x_size[0], h=x_size[2]).cuda()

    return csf

def soft_mast(self, x, x_detach, y_inter):
    """
    :param ori_cf:
    :param cf: [n, c*kernel_size, hcwc]
    :param sf: [n, c*kernel_size, hsws]
    :param mask: [hcwc, hsws]
    :return: csf [n, c*kernel_size, hcwc]
    """
    # x_m, x_v, y_m, y_v = x_m.detach(), x_v.detach(), y_m.detach(), y_v.detach()
    # x = (((x - x_m) / (x_v+1e-5).sqrt()) * (y_v+1e-5).sqrt()) + y_m
    n_cluster = 3
    x_size = x.size()
    y_inter_size = y_inter.size()
    cf, sf = self.down_sampling_feature(x_detach, y_inter)
    cf_size = cf.size()
    sf_size = sf.size()

    hc, wc = cf_size[2], cf_size[3]
    hs, ws = sf_size[2], sf_size[3]

    x = rearrange(x, 'b c h w -> c (b h w)')
    cf = rearrange(cf, 'b c h w -> c (b h w)')
    sf = rearrange(sf, 'b c h w -> c (b h w)')

    # p = self.cal_p(cf, sf)
    p = self.cal_p_attn(cf, sf)

    csf = torch.mm(p.t(), x)

    # pdb.set_trace()
    csf = rearrange(csf, 'c (b h w) -> b c h w', b=x_size[0], h=x_size[2]).cuda()

    return csf
