#include "bfs.h"

int max_deg_row(vector<vector<int>> &G)
{
    int max = 0, count = 0, row = 0;
    for (size_t i = 0; i < G.size(); i++)
    {
        count = 0;
        for (size_t j = 0; j < G[i].size(); j++)
        {
            if (G[i][j])
            {
                count++;
            }
        }
        if (count > max)
        {
            row = i;
            max = count;
        }
    }
    return row;
}
bool is_connected_from_source(vector<vector<int>> &G, vector<int> &visited, int source, vector<int> &rejection_list)
{
    queue<int> Q;
    Q.push(source);
    int current;
    while (!Q.empty())
    {
        current = Q.front();
        visited[current] = 1;
        Q.pop();
        for (size_t i = 0; i < G.size(); i++)
        {
            if (G[current][i] == 1 && visited[i] == 0)
            {
                Q.push(i);
            }
        }
    }
    bool flag = false;
    for (size_t i = 0; i < visited.size(); i++)
    {
        if (visited[i] == 0)
        {
            rejection_list.push_back(i);
            flag = true;
        }
    }
    if (flag)
    {
        return false;
    }
    return true;
}
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
}
void compute_all_paths(vector<int> parent, map<int, vector<int>> &result, int source)
{
    for (size_t i = 0; i < parent.size(); i++)
    {
        // int current = i;
        if (i != source && parent[i] != -1)
        {
            int ite = i;
            while (ite != source)
            {
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
