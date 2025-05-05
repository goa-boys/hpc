#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <queue>

using namespace std;
using namespace chrono;

struct Point {
    float x, y;
    float value; // Continuous value for regression
};

float dist(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

struct Neighbor {
    float distance;
    float value;
    float x, y;
};

// Comparator for max-heap based on distance
struct Compare {
    bool operator()(const Neighbor& a, const Neighbor& b) {
        return a.distance < b.distance; // max-heap: larger distance = higher priority
    }
};

int main() {
    int n = 1e5;
    int k_neighbors = 20;

    vector<Point> train(n);
    srand(time(0));

    // Generate random training data
    for (int i = 0; i < n; i++) {
        train[i].x = static_cast<float>(rand()) / RAND_MAX * 100;
        train[i].y = static_cast<float>(rand()) / RAND_MAX * 100;
        train[i].value = static_cast<float>(rand()) / RAND_MAX * 200;
    }

    // User input for test point
    float tx, ty;
    cout << "Enter test point (x y): ";
    cin >> tx >> ty;

    auto start = high_resolution_clock::now();

    // Compute distances in parallel
    vector<Neighbor> neighbors(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float d = dist(tx, ty, train[i].x, train[i].y);
        neighbors[i] = {d, train[i].value, train[i].x, train[i].y};
    }

    // Use max-heap to find top-k neighbors
    priority_queue<Neighbor, vector<Neighbor>, Compare> maxHeap;
    for (int i = 0; i < n; i++) {
        if (maxHeap.size() < k_neighbors) {
            maxHeap.push(neighbors[i]);
        } else if (neighbors[i].distance < maxHeap.top().distance) {
            maxHeap.pop();
            maxHeap.push(neighbors[i]);
        }
    }

    // Extract top-k neighbors
    vector<Neighbor> k_nearest;
    while (!maxHeap.empty()) {
        k_nearest.push_back(maxHeap.top());
        maxHeap.pop();
    }

    // Compute average value (regression prediction)
    float sum_values = 0.0f;
    for (const auto& neighbor : k_nearest) {
        sum_values += neighbor.value;
    }
    float pred = sum_values / k_neighbors;

    auto stop = high_resolution_clock::now();

    // Output results
    cout << "\nTop-" << k_neighbors << " Nearest Neighbors:\n";
    for (const auto& neighbor : k_nearest) {
        cout << "(" << neighbor.x << ", " << neighbor.y << ") - "
             << "Distance: " << neighbor.distance << ", Value: " << neighbor.value << "\n";
    }

    cout << "\nTest Point: (" << tx << ", " << ty << ")\n";
    cout << "Predicted Value: " << pred << "\n";
    cout << "Time taken: " << duration_cast<milliseconds>(stop - start).count() << " ms\n";

    return 0;
}
