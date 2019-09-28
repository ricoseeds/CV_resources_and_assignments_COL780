#include "bfs.h"

void bfs(vector<vector<int>> &G, int source, map<int, vector<int>> &result)
{
    queue<int> Q;
    vector<int> parent(G.size(), UNINITIALISED);
    vector<int> visited(G.size(), 0);
    Q.push(source);
    int current;
    while (!Q.empty())
    {
        current = Q.front();
        visited[current] = 1;
        // cout << current;
        Q.pop();
        for (size_t i = 0; i < G.size(); i++)
        {
            if (G[current][i] == 1 && visited[i] == 0)
            {
                Q.push(i);
                // visited[i] = 1;
                if (parent[i] == UNINITIALISED)
                {
                    parent[i] = current;
                }
            }
        }
    }
    compute_all_paths(parent, result, source);
    for (size_t i = 0; i < result.size(); i++)
    {
        cout << " <" << i << "> = { ";
        for (size_t j = 0; j < result[i].size(); j++)
        {
            cout << result[i][j] << " ";
        }
        cout << " } " << endl;
    }
}
void compute_all_paths(vector<int> parent, map<int, vector<int>> &result, int source)
{
    for (size_t i = 0; i < parent.size(); i++)
    {
        // int current = i;
        if (i != source)
        {
            int ite = i;
            while (ite != source)
            {
                cout << ite;
                result[i].push_back(ite);
                ite = parent[ite];
            }
            result[i].push_back(source);
        }
        else
        {
            result[i].push_back(-1);
        }
    }
}
