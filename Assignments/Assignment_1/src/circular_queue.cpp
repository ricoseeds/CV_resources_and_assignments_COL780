#include "circular_queue.h"

AvgCircularBuffer::AvgCircularBuffer(int size = 4)
{
    avg_flag = false;
    this->size = size;
    this->count = 0;
    CircularQ = new Point[size];
}
void AvgCircularBuffer::en_q(Point p)
{
    if (count <= size - 1)
    {
        CircularQ[count] = p;
        count = (count + 1) % size;
        if (count == size)
        {
            avg_flag = true;
        }
    }
}
float AvgCircularBuffer::avg_val_y()
{
    float accumulate_y = 0;
    if (!avg_flag)
    {
        for (size_t i = 0; i < count; i++)
        {
            accumulate_y += CircularQ[i].y;
        }
        return (accumulate_y / (float)(count));
    }
    else
    {
        for (size_t i = 0; i < size; i++)
        {
            accumulate_y += CircularQ[i].y;
        }
        return (accumulate_y / (float)size);
    }
}