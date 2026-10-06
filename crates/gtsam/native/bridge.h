#pragma once
#include <stddef.h>
#include <stdint.h>

// Fixed-layout PODs only. All arrays/matrices are row-major; no Eigen, STL,
// C++ ownership, exceptions, or allocated buffers cross the Rust boundary.
typedef struct {
  uint64_t id;
  double translation[3];
  double quaternion_xyzw[4];
} VgPose;
typedef struct {
  uint64_t from, to;
  VgPose measurement;
  double information[36];
} VgBetween;
typedef struct {
  uint64_t from, to;
  double alpha, position[3], lever[3], information[9];
  uint32_t kernel; // 0: Huber(delta), 1: eliminated switch(lambda)
  double parameter, deadband;
} VgGps;
typedef struct {
  double residual[3], squared_mahalanobis, robust_weight, optimization_weight,
      cost;
} VgGpsDiagnostic;
typedef struct {
  double initial_cost, final_cost, solve_ms;
  uint64_t iterations;
} VgReport;

#ifdef __cplusplus
extern "C" {
#endif
int visloc_gtsam_solve(const VgPose *poses, size_t pose_count,
                       const VgBetween *edges, size_t edge_count,
                       const VgGps *gps, size_t gps_count, uint64_t anchor,
                       uint32_t horizontal_anchor, VgPose *output,
                       VgGpsDiagnostic *gps_output, VgReport *report,
                       char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
