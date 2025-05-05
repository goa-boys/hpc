g++ knn.cpp -o knn -fopenmp
./knn

#include <algorithm>

void knn(int N, int K) {
    vector<pair<double, int>> distances(N);
    vector<pair<double, double>> data(N);
    vector<int> labels(N);

    for (int i = 0; i < N; i++) {
        data[i] = {rand() % 100, rand() % 100};
        labels[i] = rand() % 2; // 0 or 1
    }

    pair<double, double> test_point = {50, 50};
    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double dx = data[i].first - test_point.first;
        double dy = data[i].second - test_point.second;
        double dist = sqrt(dx * dx + dy * dy);
        distances[i] = {dist, labels[i]};
    }

    sort(distances.begin(), distances.end());

    int count0 = 0, count1 = 0;
    for (int i = 0; i < K; i++) {
        if (distances[i].second == 0) count0++;
        else count1++;
    }

    int prediction = (count1 > count0) ? 1 : 0;
    double end = omp_get_wtime();
    cout << "[KNN] Predicted Label: " << prediction << ", Time: " << end - start << " sec\n";
}
int main() {
    int N = 100000, K = 3;

    cout << "Running Linear Regression:\n";
    linear_regression(N);

    cout << "\nRunning K-Means:\n";
    k_means(N, K);

    cout << "\nRunning KNN:\n";
    knn(N, 5);

    return 0;
}
