def cluster_AdaIN(content_feats, style_feats, corr_threshold_low=0.5, corr_threshold_high=1, n_channels=32, k_clusters=5, include_content=False, power=1):
    print(content_feats.min(), content_feats.max())
    
    num_channels, content_size = content_feats.shape[1], content_feats.shape[2:]
    style_size = style_feats.shape[2:]

    content_points = content_feats.reshape(num_channels, content_size[0] * content_size[1]).T
    style_points = style_feats.reshape(num_channels, style_size[0] * style_size[1]).T

    # calculate correlation coefficient
    # corr = torch.corrcoef(torch.cat((content_points, style_points), dim=0).T)

    corr_content = torch.abs(torch.corrcoef(content_points.T))
    corr_style = torch.abs(torch.corrcoef(style_points.T))
    corr = torch.max(corr_content, corr_style)

    _, corr_channels_list = torch.topk(corr, 32, dim=1)
    print(corr_channels_list)
    print(corr_channels_list)

    # select primary channels
    corr_count = torch.count_nonzero(((corr_threshold_low < torch.abs(corr)) & (torch.abs(corr) <= corr_threshold_high)), dim=1)
    _, primary_idx = torch.topk(corr_count, n_channels, largest=True)
    print(_)
    rest_idx = find_rest(num_channels, primary_idx)

    # and split channels into the two groups
    style_primary, style_rest = split_channel(style_points, primary_idx)
    content_primary, content_rest = split_channel(content_points, primary_idx)

    # k-means clustering
    style_cluster_ids, style_cluster_centers = kmeans(X=style_primary, num_clusters=k_clusters, distance='euclidean', device='cpu')
    content_cluster_ids, content_cluster_centers = kmeans(X=content_primary, num_clusters=int(k_clusters*power), distance='euclidean', device='cpu')

    # calculate mean and std
    style_cluster_mean = []
    style_cluster_std = []

    content_cluster_mean = []
    content_cluster_std = []

    for k in range(k_clusters):
        style_idx = torch.nonzero(style_cluster_ids == k).squeeze()
        if include_content:
            style_points_in_cluster = torch.index_select(style_points, 0, style_idx)
        else:
            style_points_in_cluster = torch.index_select(style_rest, 0, style_idx)

        mean_style, std_style = calc_mean_std_2dim(style_points_in_cluster)
        style_cluster_mean.append(mean_style)
        style_cluster_std.append(std_style)

    for k in range(int(k_clusters * power)):
        content_idx = torch.nonzero(content_cluster_ids == k).squeeze()
        if include_content:
            content_points_in_cluster = torch.index_select(content_points, 0, content_idx)
        else:
            content_points_in_cluster = torch.index_select(content_rest, 0, content_idx)

        mean_content, std_content = calc_mean_std_2dim(content_points_in_cluster)
        content_cluster_mean.append(mean_content)
        content_cluster_std.append(std_content)

    # cluster matching
    if style_cluster_centers.shape[0] > 1:
        cluster_matching = kmeans_predict(content_cluster_centers, style_cluster_centers)
    else:
        cluster_matching = torch.zeros(content_cluster_centers.shape[0], dtype=torch.int32)

    # AdaIN
    for i in range(content_size[0] * content_size[1]):
        content_cluster = content_cluster_ids[i]
        style_cluster = cluster_matching[content_cluster]

        if include_content:
            normalized = (content_points[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-100)
            content_points[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]
        else:
            normalized = (content_rest[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-100)
            content_rest[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]

    if include_content:
        result = content_points
    else:
        result = merge_channel(content_primary, primary_idx, content_rest, rest_idx)
    result = result.T.reshape(content_feats.shape)

    print("success!!!")

    return result


def k_nearest_interpolation(content_feats, style_feats, corr_threshold_low=0, corr_threshold_high=0.5, n_channels=32, include_content=False, k_neighbors=5):
    num_channels, content_size = content_feats.shape[1], content_feats.shape[2:]
    style_size = style_feats.shape[2:]

    content_points = content_feats.reshape(num_channels, content_size[0] * content_size[1]).T
    style_points = style_feats.reshape(num_channels, style_size[0] * style_size[1]).T

    # calculate correlation coefficient
    corr = torch.corrcoef(torch.cat((content_points, style_points), dim=0).T)

    # select primary channels
    corr_count = torch.count_nonzero(((corr_threshold_low <= torch.abs(corr)) & (torch.abs(corr) <= corr_threshold_high)), dim=1)
    _, primary_idx = torch.topk(corr_count, n_channels, largest=True)
    rest_idx = find_rest(num_channels, primary_idx)

    # and split channels into the two groups
    style_primary, style_rest = split_channel(style_points, primary_idx)
    content_primary, content_rest = split_channel(content_points, primary_idx)

    # k-nearest interpolation
    for i in range(content_size[0] * content_size[1]):
        dist = torch.norm(style_primary - content_primary[i], dim=1)
        knn_dist, knn_idx = dist.topk(k_neighbors, largest=False)
        weight = torch.nn.functional.softmax(1 / (knn_dist), dim=0)

        if include_content:
            content_points[i] = torch.sum(style_points[knn_idx] * weight.unsqueeze(1), dim=0)
        else:
            content_rest[i] = torch.sum(style_rest[knn_idx] * weight.unsqueeze(1), dim=0)

    if include_content:
        result = content_points
    else:
        result = merge_channel(content_primary, primary_idx, content_rest, rest_idx)
    result = result.T.reshape(content_feats.shape)

    print("success!!!")

    return result


# 2022.01.12
def gaussian(dist, bandwidth):
    return torch.exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * math.sqrt(2 * math.pi))

def meanshift_step(X, bandwidth=3.5):
    for i, x in enumerate(X):
        dist = torch.norm((x - X), dim=1)
        weight = gaussian(dist, bandwidth)
        X[i] = (weight[:, None] * X).sum(0) / weight.sum()

    return X

def MeanShift_AdaIN(content, style, n_channels=16, bandwidth=3.5, corr_low=0.66, corr_high=0.99, include_primary=False):
    num_channels, content_size = content.shape[1], content.shape[2:]
    style_size = style.shape[2:]

    content_points = content.reshape(num_channels, content_size[0] * content_size[1]).T     # pixels x channels
    style_points = style.reshape(num_channels, style_size[0] * style_size[1]).T             # pixels x channels

    # calculate correlation coefficient
    corr_content = torch.abs(torch.corrcoef(content_points.T))
    corr_style = torch.abs(torch.corrcoef(style_points.T))
    corr = torch.max(corr_content, corr_style)

    # select primary channels
    corr_count = torch.count_nonzero(((corr_low < torch.abs(corr)) & (torch.abs(corr) <= corr_high)), dim=1)
    _, primary_idx = torch.topk(corr_count, n_channels, largest=True)
    rest_idx = find_rest(num_channels, primary_idx)

    # and split channels into the two groups
    style_primary, style_rest = split_channel(style_points, primary_idx)
    content_primary, content_rest = split_channel(content_points, primary_idx)

    # primary 기준으로 mean shift clustering
    bandwidth_style = estimate_bandwidth(style_primary, quantile=0.25)
    print(f"bandwidth_style: {bandwidth_style}")
    meanshift_style = MeanShift(bandwidth=bandwidth_style, cluster_all=True, n_jobs=4)

    bandwidth_content = estimate_bandwidth(content_primary, quantile=0.25)
    print(f"bandwidth_content: {bandwidth_content}")
    print()
    meanshift_content = MeanShift(bandwidth=bandwidth_content, n_jobs=4)

    style_start = time.time()
    style_cluster_ids = torch.from_numpy(meanshift_style.fit_predict(style_primary))
    style_cluster_centers = torch.from_numpy(meanshift_style.cluster_centers_)
    style_end = time.time()
    print(f"style points clustering time: {style_end - style_start} sec")
    print(f"total {style_cluster_centers.shape[0]} clusters on style image")
    print()

    content_start = time.time()
    content_cluster_ids = torch.from_numpy(meanshift_content.fit_predict(content_primary))
    content_cluster_centers = torch.from_numpy(meanshift_content.cluster_centers_)
    content_end = time.time()
    print(f"content points clustering time: {content_end - content_start} sec")
    print(f"total {content_cluster_centers.shape[0]} clusters on content image")

    # 각각의 cluster에 대해 mean, std 계산
    style_cluster_mean = []
    style_cluster_std = []
    for k in range(style_cluster_centers.shape[0]):
        style_idx = torch.nonzero(style_cluster_ids == k).squeeze()
        if include_primary:
            style_points_in_cluster = torch.index_select(style_points, 0, style_idx)
        else:
            style_points_in_cluster = torch.index_select(style_rest, 0, style_idx)

        mean_style, std_style = calc_mean_std_2dim(style_points_in_cluster)

        style_cluster_mean.append(mean_style)
        style_cluster_std.append(std_style)

    content_cluster_mean = []
    content_cluster_std = []
    for k in range(content_cluster_centers.shape[0]):
        content_idx = torch.nonzero(content_cluster_ids == k).squeeze()
        if include_primary:
            content_points_in_cluster = torch.index_select(content_points, 0, content_idx)
        else:
            content_points_in_cluster = torch.index_select(content_rest, 0, content_idx)

        mean_content, std_content = calc_mean_std_2dim(content_points_in_cluster)

        content_cluster_mean.append(mean_content)
        content_cluster_std.append(std_content)

    # cluster matching
    if style_cluster_centers.shape[0] > 1:
        cluster_matching = kmeans_predict(content_cluster_centers, style_cluster_centers)
    else:
        cluster_matching = torch.zeros(content_cluster_centers.shape[0], dtype=torch.int32)

    # cluster-wise AdaIN
    for i in range(content_size[0] * content_size[1]):
        content_cluster = content_cluster_ids[i]
        style_cluster = cluster_matching[content_cluster]

        if include_primary:
            normalized = (content_points[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-7)
            content_points[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]
        else:
            normalized = (content_rest[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-7)
            content_rest[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]

    if include_primary:
        result = content_points
    else:
        result = merge_channel(content_primary, primary_idx, content_rest, rest_idx)

    result = result.T.reshape(content.shape)
    return result


def DBSCAN_AdaIN(content, style, n_channels=16, corr_low=0.5, corr_high=0.99, include_primary=False):
    num_channels, content_size = content.shape[1], content.shape[2:]
    style_size = style.shape[2:]

    content_points = content.reshape(num_channels, content_size[0] * content_size[1]).T     # pixels x channels
    style_points = style.reshape(num_channels, style_size[0] * style_size[1]).T             # pixels x channels

    # calculate correlation coefficient
    corr_content = torch.abs(torch.corrcoef(content_points.T))
    corr_style = torch.abs(torch.corrcoef(style_points.T))
    corr = torch.max(corr_content, corr_style)

    # select primary channels
    corr_count = torch.count_nonzero(((corr_low < torch.abs(corr)) & (torch.abs(corr) <= corr_high)), dim=1)
    _, primary_idx = torch.topk(corr_count, n_channels, largest=True)
    print(_)
    rest_idx = find_rest(num_channels, primary_idx)

    # and split channels into the two groups
    style_primary, style_rest = split_channel(style_points, primary_idx)
    content_primary, content_rest = split_channel(content_points, primary_idx)

    # primary 기준으로 mean shift clustering
    DBSCAN_style = DBSCAN(eps=2)
    DBSCAN_content = DBSCAN(eps=2)

    style_cluster_ids = torch.tensor(DBSCAN_style.fit_predict(style_primary))
    content_cluster_ids = torch.tensor(DBSCAN_content.fit_predict(content_primary))

    # 각각의 cluster에 대해 mean, std 계산
    style_cluster_mean = []
    style_cluster_primary_mean = []
    style_cluster_std = []
    for k in style_cluster_ids.unique()[1:]:
        style_idx = torch.nonzero(style_cluster_ids == k).squeeze()
        if include_primary:
            style_points_in_cluster = torch.index_select(style_points, 0, style_idx)
        else:
            style_points_in_cluster = torch.index_select(style_rest, 0, style_idx)

        mean_style, std_style = calc_mean_std_2dim(style_points_in_cluster)

        style_cluster_mean.append(mean_style)
        style_cluster_std.append(std_style)

        style_primary_in_cluster = torch.index_select(style_primary, 0, style_idx)
        mean_style_primary = style_primary_in_cluster.mean(dim=0)
        style_cluster_primary_mean.append(mean_style_primary)

    content_cluster_mean = []
    content_cluster_primary_mean = []
    content_cluster_std = []
    for k in content_cluster_ids.unique()[1:]:
        content_idx = torch.nonzero(content_cluster_ids == k).squeeze()
        if include_primary:
            content_points_in_cluster = torch.index_select(content_points, 0, content_idx)
        else:
            content_points_in_cluster = torch.index_select(content_rest, 0, content_idx)

        mean_content, std_content = calc_mean_std_2dim(content_points_in_cluster)

        content_cluster_mean.append(mean_content)
        content_cluster_std.append(std_content)

        content_primary_in_cluster = torch.index_select(content_primary, 0, content_idx)
        mean_content_primary = content_primary_in_cluster.mean(dim=0)
        content_cluster_primary_mean.append(mean_content_primary)

    # 공식적으로는 cluster center가 존재하지 않기 때문에, mean을 cluster center처럼 사용한다.
    style_cluster_centers = torch.stack(style_cluster_primary_mean)
    content_cluster_centers = torch.stack(content_cluster_primary_mean)

    print("# style clusters:", style_cluster_centers.shape[0])
    print("# content clusters:", content_cluster_centers.shape[0])

    # content image의 점들 중 noise로 처리된 점들에게 가까운 cluster를 부여한다.
    noise_index = torch.nonzero(content_cluster_ids == -1).squeeze()
    num_noise = noise_index.shape[0] if noise_index.shape else 1
    if num_noise:
        noise_primary = torch.index_select(content_primary, 0, noise_index)
        print(f"total points: {content_size[0] * content_size[1]}")
        print(f"noise: {num_noise}\n")

        clustered_index = torch.nonzero(content_cluster_ids != -1).squeeze()
        clustered_primary = torch.index_select(content_primary, 0, clustered_index)

        # noise_cluster_ids = kmeans_predict(noise_primary, content_cluster_centers)
        # content_cluster_ids[noise_index] = noise_cluster_ids

    noise_cluster_ids = []
    for noise_point in noise_primary:
        dist = torch.norm(clustered_primary - noise_point.unsqueeze(0), dim=1)
        _, knn_idx = dist.topk(128, largest=False)
        # (content_size[0] * content_size[1] - noise_index.shape[0]) // 8
        knn_clusters = content_cluster_ids[knn_idx]
        noise_cluster_ids.append(torch.mode(knn_clusters, 0).values)

    content_cluster_ids[noise_index] = torch.tensor(noise_cluster_ids)

    # cluster matching
    if style_cluster_centers.shape[0] > 1 and content_cluster_centers.shape[0] > 1:
        cluster_matching = kmeans_predict(content_cluster_centers, style_cluster_centers)
    else:
        if style_cluster_centers.shape[0] == 1:
            cluster_matching = torch.zeros(content_cluster_centers.shape[0], dtype=torch.int64)
        elif content_cluster_centers.shape[0] == 1:
            dist = torch.norm(style_cluster_centers - content_cluster_centers[0], dim=1)
            _, closest_cluster = torch.topk(dist, 1, largest=False)
            cluster_matching = closest_cluster * torch.ones(content_cluster_centers.shape[0], dtype=torch.int64)

    # cluster-wise AdaIN
    for i in range(content_size[0] * content_size[1]):
        content_cluster = content_cluster_ids[i]
        style_cluster = cluster_matching[content_cluster]

        # if content_cluster == -1:
        #     continue

        if include_primary:
            normalized = (content_points[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-7)
            content_points[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]
        else:
            normalized = (content_rest[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-7)
            content_rest[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]

    if include_primary:
        result = content_points
    else:
        result = merge_channel(content_primary, primary_idx, content_rest, rest_idx)

    result = result.T.reshape(content.shape)
    return result


def HDBSCAN_AdaIN(content, style, n_channels=16, corr_low=0.66, corr_high=0.99, include_primary=False):
    num_channels, content_size = content.shape[1], content.shape[2:]
    style_size = style.shape[2:]

    content_points = content.reshape(num_channels, content_size[0] * content_size[1]).T     # pixels x channels
    style_points = style.reshape(num_channels, style_size[0] * style_size[1]).T             # pixels x channels

    # calculate correlation coefficient
    # corr = torch.corrcoef(torch.cat((content_points, style_points), dim=0).T)
    corr_content = torch.abs(torch.corrcoef(content_points.T))
    corr_style = torch.abs(torch.corrcoef(style_points.T))
    corr = torch.max(corr_content, corr_style)

    # select primary channels
    corr_count = torch.count_nonzero(((corr_low < torch.abs(corr)) & (torch.abs(corr) <= corr_high)), dim=1)
    _, primary_idx = torch.topk(corr_count, n_channels, largest=True)
    print(_)
    rest_idx = find_rest(num_channels, primary_idx)

    # and split channels into the two groups
    style_primary, style_rest = split_channel(style_points, primary_idx)
    content_primary, content_rest = split_channel(content_points, primary_idx)

    # primary 기준으로 mean shift clustering (5, 1, 5)
    HDBSCAN_style = HDBSCAN(min_cluster_size=32, min_samples=1, cluster_selection_epsilon=5)
    HDBSCAN_content = HDBSCAN(min_cluster_size=32, min_samples=1, cluster_selection_epsilon=5)

    style_cluster_ids = torch.tensor(HDBSCAN_style.fit_predict(style_primary))
    content_cluster_ids = torch.tensor(HDBSCAN_content.fit_predict(content_primary))

    # 각각의 cluster에 대해 mean, std 계산
    style_cluster_mean = []
    style_cluster_primary_mean = []
    style_cluster_std = []

    num_style_clusters = style_cluster_ids.unique().shape[0]
    if style_cluster_ids.unique()[0] == -1:
        num_style_clusters -= 1
        
    for k in range(num_style_clusters):
        style_idx = torch.nonzero(style_cluster_ids == k).squeeze()
        if include_primary:
            style_points_in_cluster = torch.index_select(style_points, 0, style_idx)
        else:
            style_points_in_cluster = torch.index_select(style_rest, 0, style_idx)

        mean_style, std_style = calc_mean_std_2dim(style_points_in_cluster)

        style_cluster_mean.append(mean_style)
        style_cluster_std.append(std_style)

        style_primary_in_cluster = torch.index_select(style_primary, 0, style_idx)
        mean_style_primary = style_primary_in_cluster.mean(dim=0)
        style_cluster_primary_mean.append(mean_style_primary)

    content_cluster_mean = []
    content_cluster_primary_mean = []
    content_cluster_std = []

    num_content_clusters = content_cluster_ids.unique().shape[0]
    if content_cluster_ids.unique()[0] == -1:
        num_content_clusters -= 1
    
    for k in range(num_content_clusters):
        content_idx = torch.nonzero(content_cluster_ids == k).squeeze()
        if include_primary:
            content_points_in_cluster = torch.index_select(content_points, 0, content_idx)
        else:
            content_points_in_cluster = torch.index_select(content_rest, 0, content_idx)

        mean_content, std_content = calc_mean_std_2dim(content_points_in_cluster)

        content_cluster_mean.append(mean_content)
        content_cluster_std.append(std_content)

        content_primary_in_cluster = torch.index_select(content_primary, 0, content_idx)
        mean_content_primary = content_primary_in_cluster.mean(dim=0)
        content_cluster_primary_mean.append(mean_content_primary)

    # 공식적으로는 cluster center가 존재하지 않기 때문에, mean을 cluster center처럼 사용한다.
    style_cluster_centers = torch.stack(style_cluster_primary_mean)
    content_cluster_centers = torch.stack(content_cluster_primary_mean)

    print("# style clusters:", style_cluster_centers.shape[0])
    print("# content clusters:", content_cluster_centers.shape[0])

    # content image의 점들 중 noise로 처리된 점들에게 가까운 cluster를 부여한다.
    noise_index = torch.nonzero(content_cluster_ids == -1).squeeze()
    num_noise = noise_index.shape[0] if noise_index.shape else 1
    if num_noise:
        noise_primary = torch.index_select(content_primary, 0, noise_index)
        print(f"total points: {content_size[0] * content_size[1]}")
        print(f"noise: {num_noise}\n")

        clustered_index = torch.nonzero(content_cluster_ids != -1).squeeze()
        clustered_primary = torch.index_select(content_primary, 0, clustered_index)

        # noise_cluster_ids = kmeans_predict(noise_primary, content_cluster_centers)
        # content_cluster_ids[noise_index] = noise_cluster_ids

        noise_cluster_ids = []
        for noise_point in noise_primary:
            dist = torch.norm(clustered_primary - noise_point, dim=1)
            _, knn_idx = dist.topk(256, largest=False)
            # (content_size[0] * content_size[1] - num_noise) // 16
            knn_clusters = content_cluster_ids[knn_idx]
            noise_cluster_ids.append(torch.mode(knn_clusters, 0).values)

        content_cluster_ids[noise_index] = torch.tensor(noise_cluster_ids)

    # cluster matching
    if style_cluster_centers.shape[0] > 1 and content_cluster_centers.shape[0] > 1:
        cluster_matching = kmeans_predict(content_cluster_centers, style_cluster_centers)
    else:
        if style_cluster_centers.shape[0] == 1:
            cluster_matching = torch.zeros(content_cluster_centers.shape[0], dtype=torch.int64)
        elif content_cluster_centers.shape[0] == 1:
            dist = torch.norm(style_cluster_centers - content_cluster_centers[0], dim=1)
            _, closest_cluster = torch.topk(dist, 1, largest=False)
            cluster_matching = closest_cluster * torch.ones(content_cluster_centers.shape[0], dtype=torch.int64)

    # cluster-wise AdaIN
    for i in range(content_size[0] * content_size[1]):
        content_cluster = content_cluster_ids[i]
        style_cluster = cluster_matching[content_cluster]

        if include_primary:
            normalized = (content_points[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-7)
            content_points[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]
        else:
            normalized = (content_rest[i] - content_cluster_mean[content_cluster]) / (content_cluster_std[content_cluster] + 1e-7)
            content_rest[i] = style_cluster_std[style_cluster] * normalized + style_cluster_mean[style_cluster]

    if include_primary:
        result = content_points
    else:
        result = merge_channel(content_primary, primary_idx, content_rest, rest_idx)

    result = result.T.reshape(content.shape)
    return result