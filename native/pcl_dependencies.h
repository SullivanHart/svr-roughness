#ifndef PCL_DEPENDENCIES_H
#define PCL_DEPENDENCIES_H

#include <pcl/pcl_config.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/bilateral.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/shot_omp.h>
#include "pcl/features/fpfh.h"
#include <pcl/io/ply_io.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/don.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/pca.h>

using namespace pcl;

void StatOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);

void EdgeDetection(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, std::string dir);

bool loadAsciCloud(std::string filename, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);

#endif
