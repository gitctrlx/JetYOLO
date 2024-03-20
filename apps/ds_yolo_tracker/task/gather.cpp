#include "gather.h"

#include <cmath>
#include <stdio.h>

using namespace std;

/**
 * Calculates the Euclidean distance between two points.
 *
 * @param p1 The first point.
 * @param p2 The second point.
 * @return The Euclidean distance between p1 and p2.
 */
double distance(const Point& p1, const Point& p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

/**
 * Performs the k-means clustering algorithm on a set of points.
 *
 * @param points The input vector of points to be clustered.
 * @param k The number of clusters to create.
 * @param maxIterations The maximum number of iterations to perform.
 * @return A vector of vectors, where each inner vector represents a cluster and contains the points belonging to that cluster.
 */
vector<vector<Point>> kMeans(const vector<Point>& points, int k, int maxIterations) {
    int n = points.size();
    vector<Point> centroids(k);
    vector<vector<Point>> clusters(k);
    for (int i = 0; i < k; i++) {
        centroids[i] = points[rand() % n];
    }
    for (int iter = 0; iter < maxIterations; iter++) {
        for (int i = 0; i < k; i++) {
            clusters[i].clear();
        }
        for (int i = 0; i < n; i++) {
            double minDist = distance(points[i], centroids[0]);
            int minIndex = 0;
            for (int j = 1; j < k; j++) {
                double d = distance(points[i], centroids[j]);
                if (d < minDist) {
                    minDist = d;
                    minIndex = j;
                }
            }
            clusters[minIndex].push_back(points[i]);
        }
        for (int i = 0; i < k; i++) {
            double sumX = 0, sumY = 0;
            int m = clusters[i].size();
            if (m == 0) {continue;}
            for (int j = 0; j < m; j++) {
                sumX += clusters[i][j].x;
                sumY += clusters[i][j].y;
            }
            centroids[i].x = sumX / m;
            centroids[i].y = sumY / m;
        }
    }
    return clusters;
}

/**
 * Calculates the standard deviation of a set of points.
 * 
 * @param points The vector of points for which to calculate the standard deviation.
 * @return The standard deviation of the points.
 */
float getStdDev(const vector<Point>& points) {
    float sumX = 0, sumY = 0;
    int n = points.size();
    for (int i = 0; i < n; i++) {
        sumX += points[i].x;
        sumY += points[i].y;
    }
    float meanX = sumX / n;
    float meanY = sumY / n;
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += pow(points[i].x - meanX, 2) + pow(points[i].y - meanY, 2);
    }
    return sqrt(sum / n);
}

/**
 * Filters out clusters based on their standard deviation and size.
 * 
 * @param clusters The input vector of clusters, where each cluster is represented by a vector of points.
 * @param threshold The threshold value for the standard deviation.
 * @param gatherPoints The output vector of clusters that pass the filtering criteria.
 */
void isGather(const vector<vector<Point>>& clusters, float threshold, vector<vector<Point>>& gatherPoints) {
    gatherPoints.clear();
    int k = clusters.size();
    for (int i = 0; i < k; i++) {
        if (clusters[i].size() == 0) {
            continue;
        }
        auto std = getStdDev(clusters[i]);
        //printf("std: %f\n", std);
        if (std < threshold && clusters[i].size() > 2) {
            gatherPoints.push_back(clusters[i]);
        }
    }
}

/**
 * Gathers points into clusters and returns the gathered points.
 *
 * @param points The input vector of points to be gathered.
 * @return A vector of vectors of points representing the gathered points.
 */
vector<vector<Point>> gather(const vector<Point>& points) {
    vector<vector<Point>> clusters = kMeans(points, 10, 100);
    vector<vector<Point>> gatherPoints;
    isGather(clusters, 80, gatherPoints);
    return gatherPoints;
}

/**
 * Represents a point in a 2D coordinate system.
 */
// struct Point {
//     int x; /**< The x-coordinate of the point. */
//     int y; /**< The y-coordinate of the point. */
// };

/**
 * Calculates the average point from a vector of points.
 *
 * @param points The vector of points.
 * @return The average point.
 */
Point averagePoint(const vector<Point>& points) {
    float sumX = 0, sumY = 0;
    int n = points.size();
    for (int i = 0; i < n; i++) {
        sumX += points[i].x;
        sumY += points[i].y;
    }
    return Point{int(sumX / n), int(sumY / n)};
}

/**
 * @brief A function that groups points based on their distance from each other.
 * 
 * This function takes a vector of points and a threshold value as input and groups the points
 * into clusters based on their distance from each other. Points that are closer to each other
 * than the threshold value are grouped together.
 * 
 * @param points The vector of points to be grouped.
 * @param threshold The maximum distance allowed between points to be considered in the same group.
 * @return A vector of vectors, where each inner vector represents a group of points.
 */
std::vector<std::vector<Point>> gather_rule(const std::vector<Point>& points, float threshold) {
    // float threshold = 200;
    std::vector<std::vector<Point>> gatherPoints;
    for (int i = 0; i < points.size(); i++) {
       for(auto& pts : gatherPoints) {
            float dist = distance(points[i], averagePoint(pts));
            // printf("dist: %f\n", dist);
            if ( dist < threshold) {
                pts.push_back(points[i]);
                break;
            }
       } 
       gatherPoints.push_back(std::vector<Point>{points[i]});
    }
    return gatherPoints;
}