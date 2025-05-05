// g++ -fopenmp assign3.cpp -o assign3
// ./assign3




#include <iostream>
#include <omp.h>
#include <climits>
using namespace std;

void min_reduction(int arr[], int n) {
  int min_value = INT_MAX;
  double start = omp_get_wtime();

  #pragma omp parallel for reduction(min: min_value)
  for (int i = 0; i < n; i++) {
    if (arr[i] < min_value) {
      min_value = arr[i];
    }
  }

  double end = omp_get_wtime();
  cout << "Minimum value: " << min_value << endl;
  cout << "Time taken (min): " << (end - start) << " seconds\n";
}

void max_reduction(int arr[], int n) {
  int max_value = INT_MIN;
  double start = omp_get_wtime();

  #pragma omp parallel for reduction(max: max_value)
  for (int i = 0; i < n; i++) {
    if (arr[i] > max_value) {
      max_value = arr[i];
    }
  }

  double end = omp_get_wtime();
  cout << "Maximum value: " << max_value << endl;
  cout << "Time taken (max): " << (end - start) << " seconds\n";
}

void sum_reduction(int arr[], int n) {
  int sum = 0;
  double start = omp_get_wtime();

  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < n; i++) {
    sum += arr[i];
  }

  double end = omp_get_wtime();
  cout << "Sum: " << sum << endl;
  cout << "Time taken (sum): " << (end - start) << " seconds\n";
}

void average_reduction(int arr[], int n) {
  int sum = 0;
  double start = omp_get_wtime();

  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < n; i++) {
    sum += arr[i];
  }

  double average = (double)sum / n;
  double end = omp_get_wtime();
  cout << "Average: " << average << endl;
  cout << "Time taken (average): " << (end - start) << " seconds\n";
}

int main() {
  int n;
  cout << "\nEnter the total number of elements: ";
  cin >> n;

  int *arr = new int[n];

  cout << "\nEnter the elements: ";
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  cout << endl;
  min_reduction(arr, n);
  max_reduction(arr, n);
  sum_reduction(arr, n);
  average_reduction(arr, n);

  delete[] arr;
  return 0;
}
