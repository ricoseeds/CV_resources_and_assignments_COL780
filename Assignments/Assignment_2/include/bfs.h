#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#define UNINITIALISED -1

using namespace std;
void bfs(vector<vector<int>> &G, int source, map<int, vector<int>> &result);
void compute_all_paths(vector<int> parent, map<int, vector<int>> &result, int source);