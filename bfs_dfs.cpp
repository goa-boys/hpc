//run karne ke liye
// g++ -fopenmp graph_parallel.cpp -o graph_parallel
// ./graph_parallel




#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <cstring>

class Graph {
public:
    int num_nodes;
    std::vector<std::vector<int>> adjacency_list;

    Graph(int nodes) : num_nodes(nodes), adjacency_list(nodes) {}

    void add_edge(int src, int dest) {
        adjacency_list[src].push_back(dest);
        adjacency_list[dest].push_back(src); // Undirected graph
    }

    void print_graph() {
        for (int i = 0; i < num_nodes; ++i) {
            std::cout << "Node " << i << ":";
            for (int neighbor : adjacency_list[i]) {
                std::cout << " " << neighbor;
            }
            std::cout << std::endl;
        }
    }

    void sequential_bfs(int start_node) {
        std::vector<bool> visited(num_nodes, false);
        std::queue<int> queue;

        visited[start_node] = true;
        queue.push(start_node);

        while (!queue.empty()) {
            int current_node = queue.front();
            queue.pop();
            std::cout << "Visited " << current_node << std::endl;

            for (int neighbor : adjacency_list[current_node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }
    }

    void parallel_bfs(int start_node) {
        std::vector<bool> visited(num_nodes, false);
        std::queue<int> queue;

        visited[start_node] = true;
        queue.push(start_node);

        while (!queue.empty()) {
            int current_node = queue.front();
            queue.pop();
            std::cout << "Visited " << current_node << std::endl;

            #pragma omp parallel for
            for (size_t i = 0; i < adjacency_list[current_node].size(); i++) {
                int neighbor = adjacency_list[current_node][i];
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    void sequential_dfs(int start_node) {
        std::vector<bool> visited(num_nodes, false);
        dfs_util(start_node, visited);
    }

    void parallel_dfs(int start_node) {
        std::vector<bool> visited(num_nodes, false);
        parallel_dfs_util(start_node, visited);
    }

private:
    void dfs_util(int node, std::vector<bool>& visited) {
        visited[node] = true;
        std::cout << "Visited " << node << std::endl;

        for (int neighbor : adjacency_list[node]) {
            if (!visited[neighbor]) {
                dfs_util(neighbor, visited);
            }
        }
    }

    void parallel_dfs_util(int node, std::vector<bool>& visited) {
        visited[node] = true;
        std::cout << "Visited " << node << std::endl;

        #pragma omp parallel for
        for (size_t i = 0; i < adjacency_list[node].size(); i++) {
            int neighbor = adjacency_list[node][i];
            if (!visited[neighbor]) {
                parallel_dfs_util(neighbor, visited);
            }
        }
    }
};

int main() {
    Graph graph(6);
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(1, 3);
    graph.add_edge(1, 4);
    graph.add_edge(2, 5);

    std::cout << "Graph adjacency list:\n";
    graph.print_graph();

    // BFS Comparison
    std::cout << "\nSequential BFS:\n";
    double start_bfs_seq = omp_get_wtime();
    graph.sequential_bfs(0);
    double end_bfs_seq = omp_get_wtime();
    std::cout << "Time (Sequential BFS): " << (end_bfs_seq - start_bfs_seq) << " seconds\n";

    std::cout << "\nParallel BFS:\n";
    double start_bfs_par = omp_get_wtime();
    graph.parallel_bfs(0);
    double end_bfs_par = omp_get_wtime();
    std::cout << "Time (Parallel BFS): " << (end_bfs_par - start_bfs_par) << " seconds\n";

    // DFS Comparison
    std::cout << "\nSequential DFS:\n";
    double start_dfs_seq = omp_get_wtime();
    graph.sequential_dfs(0);
    double end_dfs_seq = omp_get_wtime();
    std::cout << "Time (Sequential DFS): " << (end_dfs_seq - start_dfs_seq) << " seconds\n";

    std::cout << "\nParallel DFS:\n";
    double start_dfs_par = omp_get_wtime();
    graph.parallel_dfs(0);
    double end_dfs_par = omp_get_wtime();
    std::cout << "Time (Parallel DFS): " << (end_dfs_par - start_dfs_par) << " seconds\n";

    return 0;
}
