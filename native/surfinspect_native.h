#ifndef SURFINSPECT_NATIVE_H
#define SURFINSPECT_NATIVE_H

#include <cstddef>

#if defined(_WIN32) && defined(SURFINSPECT_NATIVE_EXPORTS)
#define SURFINSPECT_NATIVE_API __declspec(dllexport)
#elif defined(_WIN32)
#define SURFINSPECT_NATIVE_API __declspec(dllimport)
#else
#define SURFINSPECT_NATIVE_API
#endif

extern "C" {

struct si_config {
    double voxel_size_m;
    double mesh_grid_size_m;
    double target_edge_length_m;
    int variogram_points;
    double variogram_span_m;
    double long_cutoff_m;
    double short_cutoff_m;
    int statistical_filter;
    int statistical_mean_k;
    double statistical_stddev;
    // Appended to preserve the layout of all existing fields.
    int gaussian_mesh;
};

struct si_result {
    double sa_um;
    double sq_um;
    double svr_um;
    std::size_t input_points;
    std::size_t processed_points;
    std::size_t variogram_points;
    double* variogram_um;
    std::size_t* variogram_counts;
    std::size_t distance_count;
    double* signed_distances_mm;
};

SURFINSPECT_NATIVE_API int si_analyze_points(
    const double* xyz, std::size_t point_count,
    const si_config* config, si_result* result);
SURFINSPECT_NATIVE_API void si_free_result(si_result* result);
SURFINSPECT_NATIVE_API const char* si_last_error();

}

#endif  // SURFINSPECT_NATIVE_H
