import numpy as np
import matplotlib.pyplot as plt
import cv2
from libs import colorMap

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

# KMeans Algorithm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def clusters_distance_2(cluster1, cluster2):
    """
    Computes distance between two centroids of the two clusters
    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    @staticmethod
    def _closest_centroid(sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            print(f"len cluster: {cluster_idx} = {len(cluster)}")
            cluster_mean = np.mean(self.X[cluster], axis=0)
            print(f"cluster_mean {cluster_idx}: {cluster_mean}")
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()

    def cent(self):
        return self.centroids

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1),
                    Point(1, 0), Point(1, 1), Point(0, 1),
                    Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]

    return connects

def regionGrow(img, seeds, thresh, p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []

    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)

    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label

        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y

            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue

            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))

            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))

    return seedMark



def apply_k_means(source, k=5, max_iter=100):
    """Segment image using K-means
    Args:
        source (nd.array): BGR image to be segmented
        k (int, optional): Number of clusters. Defaults to 5.
        max_iter (int, optional): Number of iterations. Defaults to 100.
    Returns:
        segmented_image (nd.array): image segmented
        labels (nd.array): labels of every point in image
    """
    b,g,r = cv2.split(source)       # get b,g,r
    img = cv2.merge([r,g,b])     # switch it to rgb
    source = colorMap.RGB2LUV(source)
    # reshape image to points
    pixel_values = source.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # run clusters_num-means algorithm
    model = KMeans(K=k, max_iters=max_iter)
    y_pred = model.predict(pixel_values)

    centers = np.uint8(model.cent())
    y_pred = y_pred.astype(int)

    # flatten labels and get segmented image
    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(source.shape)

    return segmented_image, labels

def apply_region_growing(source: np.ndarray):

    src = np.copy(source)
    color_img = cv2.cvtColor(src, cv2.COLOR_Luv2BGR)
    img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    seeds = []
    for i in range(3):
        x = np.random.randint(0, img_gray.shape[0])
        y = np.random.randint(0, img_gray.shape[1])
        seeds.append(Point(x, y))

    # seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
    output_image = regionGrow(img_gray, seeds, 10)

    return output_image

class MeanShift:
    def __init__(self, source: np.ndarray, threshold: int):
        self.threshold = threshold
        self.current_mean_random = True
        self.current_mean_arr = []

        # Output Array
        size = source.shape[0], source.shape[1], 3
        self.output_array = np.zeros(size, dtype=np.uint8)

        # Create Feature Space
        self.feature_space = self.create_feature_space(source=source)

    def run_mean_shift(self):
        while len(self.feature_space) > 0:
            below_threshold_arr, self.current_mean_arr = self.calculate_euclidean_distance(
                current_mean_random=self.current_mean_random,
                threshold=self.threshold)

            self.get_new_mean(below_threshold_arr=below_threshold_arr)

    def get_output(self):
        return self.output_array

    @staticmethod
    def create_feature_space(source: np.ndarray):
        row = source.shape[0]
        col = source.shape[1]

        # Feature Space Array
        feature_space = np.zeros((row * col, 5))

        # Array to store RGB Values for each pixel

        counter = 0

        for i in range(row):
            for j in range(col):
                rgb_array = source[i][j]

                for k in range(5):
                    if (k >= 0) & (k <= 2):
                        feature_space[counter][k] = rgb_array[k]
                    else:
                        if k == 3:
                            feature_space[counter][k] = i
                        else:
                            feature_space[counter][k] = j
                counter += 1

        return feature_space

    def calculate_euclidean_distance(self, current_mean_random: bool, threshold: int):
        """
        calculate the Euclidean distance of all the other pixels in M with the current mean.
        """

        below_threshold_arr = []

        # selecting a random row from the feature space and assigning it as the current mean
        if current_mean_random:
            current_mean = np.random.randint(0, len(self.feature_space))
            self.current_mean_arr = self.feature_space[current_mean]

        for f_indx, feature in enumerate(self.feature_space):
            # Finding the euclidean distance of the randomly selected row i.e. current mean with all the other rows
            ecl_dist = euclidean_distance(self.current_mean_arr, feature)

            # Checking if the distance calculated is within the threshold. If yes taking those rows and adding
            # them to a list below_threshold_arr
            if ecl_dist < threshold:
                below_threshold_arr.append(f_indx)

        return below_threshold_arr, self.current_mean_arr

    def get_new_mean(self, below_threshold_arr: list):
        iteration = 0.01

        # For all the rows found and placed in below_threshold_arr list, calculating the average of
        # Red, Green, Blue and index positions.
        print(f"features shape: {self.feature_space.shape}")
        mean_r = np.mean(self.feature_space[below_threshold_arr][:, 0])
        mean_g = np.mean(self.feature_space[below_threshold_arr][:, 1])
        mean_b = np.mean(self.feature_space[below_threshold_arr][:, 2])
        mean_i = np.mean(self.feature_space[below_threshold_arr][:, 3])
        mean_j = np.mean(self.feature_space[below_threshold_arr][:, 4])

        # Finding the distance of these average values with the current mean and comparing it with iter
        mean_e_distance = (euclidean_distance(mean_r, self.current_mean_arr[0]) +
                           euclidean_distance(mean_g, self.current_mean_arr[1]) +
                           euclidean_distance(mean_b, self.current_mean_arr[2]) +
                           euclidean_distance(mean_i, self.current_mean_arr[3]) +
                           euclidean_distance(mean_j, self.current_mean_arr[4]))

        # If less than iter, find the row in below_threshold_arr that has i, j nearest to mean_i and mean_j
        # This is because mean_i and mean_j could be decimal values which do not correspond
        # to actual pixel in the Image array.
        if mean_e_distance < iteration:
            new_arr = np.zeros((1, 3))
            new_arr[0][0] = mean_r
            new_arr[0][1] = mean_g
            new_arr[0][2] = mean_b

            # When found, color all the rows in below_threshold_arr with
            # the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
            for i in range(len(below_threshold_arr)):
                m = int(self.feature_space[below_threshold_arr[i]][3])
                n = int(self.feature_space[below_threshold_arr[i]][4])
                self.output_array[m][n] = new_arr

                # Also now don't use those rows that have been colored once.
                self.feature_space[below_threshold_arr[i]][0] = -1

            self.current_mean_random = True
            new_d = np.zeros((len(self.feature_space), 5))
            counter_i = 0

            for i in range(len(self.feature_space)):
                if self.feature_space[i][0] != -1:
                    new_d[counter_i][0] = self.feature_space[i][0]
                    new_d[counter_i][1] = self.feature_space[i][1]
                    new_d[counter_i][2] = self.feature_space[i][2]
                    new_d[counter_i][3] = self.feature_space[i][3]
                    new_d[counter_i][4] = self.feature_space[i][4]
                    counter_i += 1

            self.feature_space = np.zeros((counter_i, 5))

            counter_i -= 1
            for i in range(counter_i):
                self.feature_space[i][0] = new_d[i][0]
                self.feature_space[i][1] = new_d[i][1]
                self.feature_space[i][2] = new_d[i][2]
                self.feature_space[i][3] = new_d[i][3]
                self.feature_space[i][4] = new_d[i][4]

        else:
            self.current_mean_random = False
            self.current_mean_arr[0] = mean_r
            self.current_mean_arr[1] = mean_g
            self.current_mean_arr[2] = mean_b
            self.current_mean_arr[3] = mean_i
            self.current_mean_arr[4] = mean_j

def apply_mean_shift(source: np.ndarray, threshold: int = 60):
    src = np.copy(source)
    ms = MeanShift(source=src, threshold=threshold)
    ms.run_mean_shift()
    output = ms.get_output()

    return output

clusters_list = []
cluster = {}
centers = {}

def initial_clusters( points):
  
    global initial_k
    groups = {}
    d = int(256 / initial_k)
    for i in range(initial_k):
        j = i * d
        groups[(j, j, j)] = []
    for i, p in enumerate(points):
        go = min(groups.keys(), key=lambda c: np.sqrt(np.sum((p - c) ** 2)))
        groups[go].append(p)
    return [g for g in groups.values() if len(g) > 0]

def fit(points):
    global clusters_list
    clusters_list = initial_clusters(points)

    while len(clusters_list) > clusters_number:
        cluster1, cluster2 = min(
            [(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
            key=lambda c: clusters_distance_2(c[0], c[1]))

        clusters_list = [c for c in clusters_list if c != cluster1 and c != cluster2]

        merged_cluster = cluster1 + cluster2

        clusters_list.append(merged_cluster)

    global cluster 
    for cl_num, cl in enumerate(clusters_list):
        for point in cl:
            cluster[tuple(point)] = cl_num

    global centers 
    for cl_num, cl in enumerate(clusters_list):
        centers[cl_num] = np.average(cl, axis=0)

def predict_cluster(point):
    global cluster
  
    return cluster[tuple(point)]

def predict_center( point):
  
    point_cluster_num = predict_cluster(point)
    center = centers[point_cluster_num]
    return center

# input_image = cv2.imread('../images/seg-image.png')

def apply_agglomerative_clustering(source, number_of_clusters,initial_number_of_clusters):
    global clusters_number
    global initial_k
    clusters_number = number_of_clusters
    initial_k = initial_number_of_clusters 
    src = np.copy(source.reshape((-1, 3)))

    fit(src)

    output_image = [[predict_center(list(src)) for src in row] for row in source]
    output_image = np.array(output_image, np.uint8)
    return output_image