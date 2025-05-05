g++ kmean.cpp -o kmean -fopenmp
./kmean

#include <cmath>

void k_means(int N, int K) {
    vector<pair<double, double>> points(N);
    vector<pair<double, double>> centroids(K);
    vector<int> labels(N);
    srand(time(0));

    for (int i = 0; i < N; i++) {
        points[i] = {rand() % 100, rand() % 100};
    }
    for (int i = 0; i < K; i++) {
        centroids[i] = {rand() % 100, rand() % 100};
    }

    double start = omp_get_wtime();
    for (int it = 0; it < 10; it++) {
        // Assign labels
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double min_dist = 1e9;
            for (int j = 0; j < K; j++) {
                double dx = points[i].first - centroids[j].first;
                double dy = points[i].second - centroids[j].second;
                double dist = dx * dx + dy * dy;
                if (dist < min_dist) {
                    min_dist = dist;
                    labels[i] = j;
                }
            }
        }

        // Update centroids
        vector<double> sum_x(K, 0), sum_y(K, 0);
        vector<int> count(K, 0);
        for (int i = 0; i < N; i++) {
            int label = labels[i];
            sum_x[label] += points[i].first;
            sum_y[label] += points[i].second;
            count[label]++;
        }
        for (int j = 0; j < K; j++) {
            if (count[j] != 0) {
                centroids[j] = {sum_x[j] / count[j], sum_y[j] / count[j]};
            }
        }
    }
    double end = omp_get_wtime();
    cout << "[K-Means] Time: " << end - start << " sec\n";
}
