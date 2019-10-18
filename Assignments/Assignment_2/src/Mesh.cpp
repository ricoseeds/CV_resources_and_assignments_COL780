#include "Mesh.h"

bool Mesh::loadOBJ(const std::string &filename)
{
    if (filename.find(".obj") != std::string::npos)
    {
        std::ifstream fin(filename, std::ios::in);
        if (!fin)
        {
            std::cerr << "Cannot open " << filename << std::endl;
            return false;
        }

        std::cout << "Loading OBJ file " << filename << " ..." << std::endl;
        std::string lineBuffer;
        while (std::getline(fin, lineBuffer))
        {
            std::stringstream ss(lineBuffer);
            std::string cmd;
            ss >> cmd;
            if (cmd == "v")
            {
                Vec3d vertex;
                int dim = 0;
                while (dim < 3 && ss >> vertex[dim])
                    dim++;
                vertices.push_back(vertex);
            }
            else if (cmd == "f")
            {
                std::string faceData;
                Vec3i index;
                int u = 0;
                while (ss >> faceData)
                {
                    std::vector<std::string> data = split(faceData, "/");

                    if (data[0].size() > 0)
                    {
                        sscanf(data[0].c_str(), "%d", &index[u++]);
                    }
                }
                faces.push_back(index);
            }
        }
    }
    return true;
}

std::vector<std::string> split(std::string s, std::string t)
{
    std::vector<std::string> res;
    while (1)
    {
        int pos = s.find(t);
        if (pos == -1)
        {
            res.push_back(s);
            break;
        }
        res.push_back(s.substr(0, pos));
        s = s.substr(pos + 1, s.size() - pos - 1);
    }
    return res;
}
