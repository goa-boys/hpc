#include <iostream>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace chrono;

int main() {
    const int n = 1e5;
    float* x = new float[n];
    float* y = new float[n];

    // Create training data: y = 2x + 3
    for (int i = 0; i < n; ++i) {
        x[i] = (i + 1) / float(n);  // Normalized input
        y[i] = 2 * x[i] + 3;
    }

    float w = 0, b = 0, lr = 0.01;
    float dw, db;

    auto start = high_resolution_clock::now();  // Start time

    for (int epoch = 0; epoch < 2000; ++epoch) {
        dw = 0.0f;
        db = 0.0f;
        float loss = 0.0f;

        #pragma omp parallel for reduction(+:dw, db, loss)
        for (int i = 0; i < n; ++i) {
            float y_pred = w * x[i] + b;
            float error = y_pred - y[i];
            dw += 2 * x[i] * error / n;
            db += 2 * error / n;
            loss += error * error / n;
        }

        w -= lr * dw;
        b -= lr * db;

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << ": w = " << w << ", b = " << b << ", loss = " << loss << endl;
        }
    }

    auto stop = high_resolution_clock::now();  // End time
    double duration = duration_cast<milliseconds>(stop - start).count();

    cout << "\nFinal Learned Parameters: w = " << w << ", b = " << b << endl;
    cout << "Training Time: " << duration << " ms" << endl;

    delete[] x;
    delete[] y;

    return 0;
}
