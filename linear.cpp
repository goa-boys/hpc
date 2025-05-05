g++ linear.cpp -o linear -fopenmp
./linear

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

void linear_regression(int N) {
    vector<double> x(N), y(N);
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    srand(time(0));

    for (int i = 0; i < N; i++) {
        x[i] = rand() % 100;
        y[i] = 2 * x[i] + rand() % 10;  // y = 2x + noise
    }

    // Sequential
    double start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    double m = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x * sum_x);
    double b = (sum_y - m * sum_x) / N;
    double end = omp_get_wtime();
    cout << "[Linear Regression] Sequential Time: " << end - start << " sec\n";

    // Parallel
    sum_x = sum_y = sum_xy = sum_xx = 0;
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_x, sum_y, sum_xy, sum_xx)
    for (int i = 0; i < N; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }
    m = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x * sum_x);
    b = (sum_y - m * sum_x) / N;
    end = omp_get_wtime();
    cout << "[Linear Regression] Parallel Time: " << end - start << " sec\n";
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
