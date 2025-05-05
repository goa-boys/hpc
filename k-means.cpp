#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace chrono;

struct Point {
    float x, y;
    int cluster;
};

float dist(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

int main() {
    int n = 1e5, k = 5;
    vector<Point> pts(n);
    vector<float> cx(k), cy(k);

    srand(time(0));
    for (int i = 0; i < n; i++) {
        pts[i].x = rand() % 100;
        pts[i].y = rand() % 100;
        pts[i].cluster = rand() % k;
    }
    for (int i = 0; i < k; i++) {
        cx[i] = rand() % 100;
        cy[i] = rand() % 100;
    }

    auto start = high_resolution_clock::now();

    for (int it = 0; it < 20; it++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            float dmin = 1e9;
            for (int j = 0; j < k; j++) {
                float d = dist(pts[i].x, pts[i].y, cx[j], cy[j]);
                if (d < dmin) {
                    dmin = d;
                    pts[i].cluster = j;
                }
            }
        }

        vector<float> sx(k, 0), sy(k, 0);
        vector<int> count(k, 0);
        for (int i = 0; i < n; i++) {
            int c = pts[i].cluster;
            sx[c] += pts[i].x;
            sy[c] += pts[i].y;
            count[c]++;
        }

        for (int j = 0; j < k; j++) {
            if (count[j]) {
                cx[j] = sx[j] / count[j];
                cy[j] = sy[j] / count[j];
            }
        }
    }

    auto stop = high_resolution_clock::now();
    cout << "Time taken: " << duration_cast<milliseconds>(stop - start).count() << " ms\n";

    // Final centroids
    cout << "\nFinal centroids:\n";
    for (int i = 0; i < k; i++) {
        cout << "Cluster " << i << ": (" << cx[i] << ", " << cy[i] << ")\n";
    }

    // Print a few sample points
    cout << "\nSample points:\n";
    for (int i = 0; i < 10; i++) {
        cout << "(" << pts[i].x << ", " << pts[i].y << ") => Cluster " << pts[i].cluster << "\n";
    }
}
