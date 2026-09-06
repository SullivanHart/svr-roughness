#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Force full re-link 2026-08-28 13:33:30
#include "surfinspect_native.h"
#include "cloud_viewer.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

void copy_result(double sa_um, double sq_um, double svr_um, std::size_t input_points,
                 const std::vector<double>& variogram, const std::vector<double>& distances,
                 si_result* result) {
    result->sa_um = sa_um;
    result->sq_um = sq_um;
    result->svr_um = svr_um;
    result->input_points = input_points;
    result->processed_points = distances.size();
    result->variogram_points = variogram.size();
    result->variogram_um = new double[result->variogram_points];
    result->variogram_counts = new std::size_t[result->variogram_points];
    std::copy(variogram.begin(), variogram.end(), result->variogram_um);
    std::fill(result->variogram_counts,
              result->variogram_counts + result->variogram_points, 0);
    result->distance_count = distances.size();
    result->signed_distances_mm = new double[result->distance_count];
    std::copy(distances.begin(), distances.end(),
              result->signed_distances_mm);
}
}

extern "C" int si_analyze_points(const double* xyz, std::size_t point_count,
                                  const si_config* config, si_result* result) {
    if (!result || (!xyz && point_count != 0) || !config) {
        last_error = "Invalid native API arguments";
        return 1;
    }
    *result = {};
    try {
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZRGBA>);
        cloud->reserve(point_count);
        for (std::size_t i = 0; i < point_count; ++i) {
            pcl::PointXYZRGBA point;
            point.x = static_cast<float>(xyz[i * 3]);
            point.y = static_cast<float>(xyz[i * 3 + 1]);
            point.z = static_cast<float>(xyz[i * 3 + 2]);
            cloud->push_back(point);
        }
        if (cloud->size() < 20) {
            throw std::runtime_error("SurfInspect native core requires at least 20 points");
        }

        pcl::PointXYZRGBA centroid;
        CenterPointCloudAndAdjustUnits(cloud, centroid, true, false);
        VoxelGridDownsample(cloud, config->voxel_size_m);

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr combined(
            new pcl::PointCloud<pcl::PointXYZRGBA>);
        std::vector<double> variogram(config->variogram_points);
        std::vector<std::vector<double>> variogram_matrix;
        std::vector<double> svr_points;
        std::vector<double> distances;
        std::vector<double> metrics;
        bool retflag = false;
        auto begin = std::chrono::steady_clock::now();
        std::string dir;
        std::string file;
        if (config->gaussian_mesh != 0) {
            boost::shared_ptr<pcl::PolygonMesh> inputMesh;
            createDenseGridPointCloud(cloud, config->voxel_size_m, inputMesh, false);
            filterShortLong(cloud, config->voxel_size_m, false, dir, file,
                            static_cast<float>(config->short_cutoff_m),
                            static_cast<float>(config->long_cutoff_m));
            roughnessCalculation(cloud, config->voxel_size_m, config->variogram_points,
                                 config->variogram_span_m, variogram, metrics, 2, true);
            if (metrics.size() < 3) {
                throw std::runtime_error("SurfInspect grid roughness calculation failed");
            }
            distances.resize(cloud->size());
            for (std::size_t i = 0; i < cloud->size(); ++i) {
                distances[i] = cloud->points[i].z;
            }
        } else {
            const int status = prepareVariogramCalculation(
                cloud, combined, dir, file, config->statistical_filter != 0, false,
                false, config->voxel_size_m, config->mesh_grid_size_m, centroid,
                true, false, config->target_edge_length_m, begin, variogram,
                config->variogram_points, config->variogram_span_m,
                variogram_matrix, svr_points, distances, retflag,
                false, false, metrics, 2, true,
                config->long_cutoff_m, config->short_cutoff_m);
            if (status != 0 || retflag || metrics.size() <= 2) {
                throw std::runtime_error("SurfInspect roughness calculation failed");
            }
        }

        double sa = (metrics.size() > 0) ? metrics[0] : 0.0;
        double sq = (metrics.size() > 1) ? metrics[1] : 0.0;
        double svr = (metrics.size() > 2) ? metrics[2] : 0.0;
        if ((sa == 0.0 || sq == 0.0) && !distances.empty()) {
            double sum_abs = 0.0;
            double sum_sq = 0.0;
            for (double d : distances) {
                sum_abs += std::abs(d);
                sum_sq += d * d;
            }
            sa = (sum_abs / distances.size()) * 1000000.0;
            sq = std::sqrt(sum_sq / distances.size()) * 1000000.0;
        }
        result->sa_um = sa;
        result->sq_um = sq;
        result->svr_um = svr;
        result->input_points = point_count;
        result->processed_points = cloud->size();
        result->variogram_points = variogram.size();
        result->variogram_um = new double[variogram.size()];
        result->variogram_counts = new std::size_t[variogram.size()];
        for (std::size_t i = 0; i < variogram.size(); ++i) {
            result->variogram_um[i] = variogram[i] * 1000.0;
            result->variogram_counts[i] = 1;
        }
        result->distance_count = distances.size();
        result->signed_distances_mm = new double[distances.size()];
        for (std::size_t i = 0; i < distances.size(); ++i) {
            result->signed_distances_mm[i] = distances[i] * 1000.0;
        }

        if (cloud->width > 1 && cloud->height > 1 && cloud->width * cloud->height == cloud->size()) {
            result->grid_width = cloud->width;
            result->grid_height = cloud->height;
            result->grid_origin_x_mm = cloud->points[0].x * 1000.0;
            result->grid_origin_y_mm = cloud->points[0].y * 1000.0;
            result->grid_z_mm = new double[cloud->size()];
            for (std::size_t i = 0; i < cloud->size(); ++i) {
                // If it's a hole filler flag or huge number, output nan
                if (std::abs(cloud->points[i].z) > 10.0) {
                    result->grid_z_mm[i] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    result->grid_z_mm[i] = cloud->points[i].z * 1000.0;
                }
            }

            // High-speed parallel local S_VR spatial variogram map (instant in C++)
            result->grid_svr_um = new double[cloud->size()];
            const int w = static_cast<int>(cloud->width);
            const int h = static_cast<int>(cloud->height);
            const double max_dist_m = config->variogram_points * config->variogram_span_m;
            const double max_dist_sq = max_dist_m * max_dist_m;
            const int radius_cells = (config->voxel_size_m > 0.0)
                ? static_cast<int>(std::ceil(max_dist_m / config->voxel_size_m))
                : 25;
            const double pitch_m = config->voxel_size_m;

            struct LocalOffset {
                int dr;
                int dc;
            };
            std::vector<LocalOffset> active_offsets;
            for (int dr = -radius_cells; dr <= radius_cells; ++dr) {
                for (int dc = -radius_cells; dc <= radius_cells; ++dc) {
                    double dist_sq = (dr * dr + dc * dc) * pitch_m * pitch_m;
                    if (dist_sq > 0.0 && dist_sq <= max_dist_sq) {
                        active_offsets.push_back({ dr, dc });
                    }
                }
            }

            tbb::parallel_for(0, h, [&](int r) {
                const bool safe_r = (r >= radius_cells && r < h - radius_cells);
                for (int c = 0; c < w; ++c) {
                    const int idx = r * w + c;
                    const double center_z = cloud->points[idx].z;
                    if (std::abs(center_z) > 10.0 || std::isnan(center_z)) {
                        result->grid_svr_um[idx] = std::numeric_limits<double>::quiet_NaN();
                        continue;
                    }

                    double sum_sq = 0.0;
                    std::size_t count = 0;

                    if (safe_r && c >= radius_cells && c < w - radius_cells) {
                        // Fast interior branch: no bounds checks needed
                        for (const auto& off : active_offsets) {
                            const int n_idx = (r + off.dr) * w + (c + off.dc);
                            const double nz = cloud->points[n_idx].z;
                            if (std::abs(nz) <= 10.0 && !std::isnan(nz)) {
                                const double diff_um = (center_z - nz) * 1000000.0;
                                sum_sq += diff_um * diff_um;
                                count++;
                            }
                        }
                    } else {
                        // Boundary branch: check perimeter limits
                        for (const auto& off : active_offsets) {
                            const int nr = r + off.dr;
                            const int nc = c + off.dc;
                            if (nr >= 0 && nr < h && nc >= 0 && nc < w) {
                                const int n_idx = nr * w + nc;
                                const double nz = cloud->points[n_idx].z;
                                if (std::abs(nz) <= 10.0 && !std::isnan(nz)) {
                                    const double diff_um = (center_z - nz) * 1000000.0;
                                    sum_sq += diff_um * diff_um;
                                    count++;
                                }
                            }
                        }
                    }

                    if (count > 0) {
                        result->grid_svr_um[idx] = std::sqrt(sum_sq / count);
                    } else {
                        result->grid_svr_um[idx] = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            });
        } else {
            result->grid_width = 0;
            result->grid_height = 0;
            result->grid_origin_x_mm = 0;
            result->grid_origin_y_mm = 0;
            result->grid_z_mm = nullptr;
            result->grid_svr_um = nullptr;
        }

        return 0;
    } catch (const std::exception& error) {
        si_free_result(result);
        last_error = error.what();
        return 1;
    } catch (...) {
        si_free_result(result);
        last_error = "Unknown Cloud-Viewer native error";
        return 1;
    }
}

extern "C" void si_free_result(si_result* result) {
    if (!result) return;
    delete[] result->variogram_um;
    delete[] result->variogram_counts;
    delete[] result->signed_distances_mm;
    delete[] result->grid_z_mm;
    delete[] result->grid_svr_um;
    *result = {};
}

extern "C" const char* si_last_error() {
    return last_error.c_str();
}
