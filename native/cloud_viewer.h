#ifndef PCL_CLOUD_VIEWER_H_
#define PCL_CLOUD_VIEWER_H_

#include "types.h"
#include "pcl_dependencies.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <iterator>
#include <memory>

//Dependencies
//PCL
//CGAL
//TBB
//Boost
//Eigen

#include <CGAL/Scale_space_reconstruction_3/Jet_smoother.h>
#include <CGAL/Scale_space_reconstruction_3/Advancing_front_mesher.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/wlop_simplify_and_regularize_point_set.h>
#include <CGAL/bilateral_smooth_point_set.h>
#include <CGAL/Kernel/global_functions.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/distance.h>
#include <CGAL/Scale_space_surface_reconstruction_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Epick_d.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/Surface_mesh/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/boost/graph/copy_face_graph.h>

#include <boost/variant.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/iterator/function_output_iterator.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_arena.h"
#include <boost/bind/bind.hpp> // Ensure this is included

using namespace tbb;
using namespace Eigen;
using namespace std;
using namespace boost::numeric::ublas;

extern double parallelDistanceTime;
extern double loop1Time;
extern double loop2Time;
extern double loop3Time;
extern double loop4Time;
extern double loop5Time;
extern double loop6Time;

extern int user_data;
extern bool saveClouds;

extern PointList PointsGlobal;

extern pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudGLOBAL;
extern float MeanGrey;
extern bool GageRR;
extern std::string Operator;
extern std::string Casting;
extern std::string Run;
extern std::string dir;
extern std::string file;

Eigen::Matrix3f Inverse2(Eigen::Matrix3f matIn);

void TransformBack(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, Eigen::Matrix3f matIn, Eigen::Vector4f centroid = Eigen::Vector4f::Zero());

Eigen::Vector4f Transformation(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, Eigen::Matrix3f eig_vec);

Eigen::Matrix3f PCAcalc(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr);

void passfilter(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
	const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputcloudptr, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& outputcloudptr);

void passfilter2DGrid(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampleValue);

void passfilter2D(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampleValue);

void GridFilter(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr & inputptr, double downsampleValue);

//Distance defenition
struct Distance {
	const Tree* inputTree;
	double* output;
	void operator()(const blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); ++i) {
			Point point_query = PointsGlobal[i].first;
			output[i] = sqrt(inputTree->squared_distance(point_query));
		}
	}
};

void  ParallelDistance(std::vector<double>& output, PointList input, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointsOnSurface, Tree* inputTree,
	size_t n/*, CGAL::Side_of_triangle_mesh<Polyhedron, Kernel> inside*/);

//CGAL struct necessary for meshing
struct halfedge2edge
{
	halfedge2edge(const Mesh2& m, std::vector<edge_descriptor>& edges)
		: m_mesh(m), m_edges(edges)
	{}
	void operator()(const halfedge_descriptor& h) const
	{
		m_edges.push_back(edge(h, m_mesh));
	}
	const Mesh2& m_mesh;
	std::vector<edge_descriptor>& m_edges;
};

struct halfedge2edge2
{
	halfedge2edge2(const Polyhedron& m, std::vector<edge_descriptor2>& edges)
		: m_mesh(m), m_edges(edges)
	{}
	void operator()(const halfedge_descriptor2& h) const
	{
		m_edges.push_back(edge(h, m_mesh));
	}
	const Polyhedron& m_mesh;
	std::vector<edge_descriptor2>& m_edges;
};

double calcVariogramFromPCParallel(std::vector<double>& SvrPoint2, const PointCloud<PointXYZRGBA>::Ptr& cloud,
	std::vector<double> dist, std::vector<double>& varVec, std::vector<double>& sumVec, std::vector<double>& CtrVec, int PointsOnVariogram = 5,
	double span = 0.001);

std::vector<double> calcVariogramFromPCParallel_local(PointList points2, std::vector<double>& varVec, std::vector<double>& sumVec,
	std::vector<double>& CtrVec, int PointsOnVariogram = 5, double span = 0.001, bool exactDistance = false);

void downsampleCGAL(PointList& points, double cell_size_grid_simplify_point_set = 0.0002);

void checkNormalOrientation(PointList& points, pcl::PointXYZRGBA Centroid);

void estimateNormalsCGAL(PointList& points);

void createMeshBezier(PointList& points, int loopCounter, string dir, string file, double target_edge_length);

Polyhedron createDenseMesh(Polyhedron output_mesh_buffer, PointList& points, int loopCounter, string dir, string file);

Polyhedron remesh(Polyhedron& mesh2, int loopCounter, double target_edge_length, string dir, string file, bool secondRun = false);

void colorPcBasedOnDist(std::vector<double>& dist2, PointCloud<PointXYZRGBA>::Ptr& cloud);

void colorPcBasedOnDist(std::vector<double>& dist2, PointCloud<PointXYZRGBA>::Ptr& cloud, string dirFile, int number);

void colorPcBasedOnAbnormalities(std::vector<double>& SvrPointCombined, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudRGBA, double SvrFinal, std::vector<double>& dist2Combined,
	int abnormalitySelection);

void colorPcBasedOnLocalVariogram(std::vector<double>& SvrPoint, PointCloud<PointXYZRGBA>::Ptr& cloudRGBA);

void Prep(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer, const pcl::PointXYZRGBA& outCentroid2, PointList& points, PointList& points2, int j,
	std::string& dir, std::string& file, bool statFilter, bool GridFilterBool, bool removeEdgesAfterMeshing, double downsample, double cell_size_grid_simplify_point_set);

void clusterPrep(int j, std::vector<pcl::PointIndices>::const_iterator& it, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& edgesRemoved, /*std::stringstream& ss,*/ std::string& dir,
	std::string& file, bool statFilter, bool GridFilterBool, bool removeEdgesAfterMeshing, double downsample, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer,
	PointList& points2, double cell_size_grid_simplify_point_set, PointList& points, const pcl::PointXYZRGBA& outCentroid2);

void outputClusterCommandLine(std::vector<pcl::PointIndices>& cluster_indices, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr edgesRemoved);

MatrixXf crateGaussFilterKernel(MatrixXf GKernel, double cutoffWaveLength, int KernelSize, bool high = true);

void applyFilterKernelToPc(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, MatrixXf GKernelLow, MatrixXf GKernelHigh, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudXYZ, int KernelSize);

void applyGaussianFilterToCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, MatrixXf GKernel, int KernelSize, bool HighPass, bool cropCloud, string dir, string file);

int determineKernelSizeFromCutOffWithAccuracyCheck(double cutoffWaveLength);

int determineKernelSizeFromCutOff(double cutoffWaveLength, float downsampleValue);

Polyhedron gcreateGaussGridMesh(PointList& points, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudXYZ, Polyhedron output_mesh_buffer, int loopCounter,
	string dir, string file, double longWavelengthCutoff, double shortWavelengthCutoff);

std::vector<double> calcDistFromLocalGauss(PointList& points, PointCloud<PointXYZRGBA>::Ptr& cloud2, std::vector<double>& dist, size_t n, std::vector<double>& varVec,
	std::vector<double>& sumVec, std::vector<double>& CtrVec, int PointsOnVariogram, double span, string dir, string file, int j,
	double shortWavelengthCutoff, double longWavelengthCutoff);

int globalMeshingVariogram(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer, PointList& points, PointList& points2,
	bool bezier, int j, std::string& dir, std::string& file, double target_edge_length, std::vector<double>& dist,
	std::chrono::steady_clock::time_point& begin, std::vector<double>& SvrPoint, /*double* Svr,*/ std::vector<double>& varVec, std::vector<double>& sumVec, std::vector<double>& CtrVec,
	int PointsOnVariogram, double span, bool& retflag, std::vector<double>& SaSqSvr, bool GaussDistance, bool gaussGrd,
	int RoughnessParameterSaSqSvr, bool OverWriteParameters, double longWavelengthCutoff, double shortWavelengthCutoff);

void clusterFinish(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBACombined,
	std::vector<std::vector<double>>& sumMatrix, std::vector<double>& sumVec, std::vector<std::vector<double>>& CtrMatrix, std::vector<double>& CtrVec,
	std::vector<std::vector<double>>& varMatrix, std::vector<double>& varVec, std::vector<double>& SvrPoint, int& j,
	std::vector<double>& SvrPointCombined, std::vector<double>& dist2Combined, std::vector<double>& dist, int RoughnessParameterSaSqSvr);

void createResultTable(std::string& dir, std::string& file, std::vector<std::vector<double>>& varMatrix, double downsample,
	double cell_size_grid_simplify_point_set, int PointsOnVariogram, double span, double target_edge_length, std::vector<double>& varVec,
	double& SvrFinal, double time, int PCsize, std::vector<double> SaSqSvr, int RoughnessParameterSaSqSvr);

void clusterPc(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& edgesRemoved, std::vector<pcl::PointIndices>& cluster_indices);

int prepareVariogramCalculation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudRGBACombined,
	std::string& dir, std::string& file, bool statFilter, bool GridFilterBool, bool removeEdgesAfterMeshing, double downsample,
	double cell_size_grid_simplify_point_set, pcl::PointXYZRGBA& outCentroid2, bool globalMeshing, bool bezier, double target_edge_length,
	std::chrono::steady_clock::time_point& begin, /*double* Svr,*/ std::vector<double>& varVec, int PointsOnVariogram, double span, std::vector<std::vector<double>>& varMatrix,
	std::vector<double>& SvrPointCombined, std::vector<double>& dist2Combined, bool& retflag, bool GaussDistance, bool gaussGrd, std::vector<double>& SaSqSvr,
	int RoughnessParameterSaSqSvr, bool OverWriteParameters, double longWavelengthCutoff, double shortWavelengthCutoff);

int RoughnessCalcCluster(std::vector<pcl::PointIndices>& cluster_indices, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& edgesRemoved, std::string& dir, std::string& file,
	bool statFilter, bool GridFilterBool, bool removeEdgesAfterMeshing, double downsample, double cell_size_grid_simplify_point_set,
	pcl::PointXYZRGBA& outCentroid2, bool globalMeshing, bool bezier, double target_edge_length, std::chrono::steady_clock::time_point& begin, /*double* Svr,*/
	std::vector<double>& varVec, int PointsOnVariogram, double span, std::vector<std::vector<double>>& varMatrix, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudRGBACombined,
	std::vector<double>& SvrPointCombined, std::vector<double>& dist2Combined, bool& retflag, bool GaussDistance, bool gaussGrd, std::vector<double>& SaSqSvr, int RoughnessParameterSaSqSvr,
	bool OverWriteParameters, double longWavelengthCutoff, double shortWavelengthCutoff);

bool LoadInputFromFile(std::string& fileEnding, std::string& filename, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, boost::shared_ptr< ::pcl::PolygonMesh>  inputMesh, bool stlInput);

void CenterPointCloudAndAdjustUnits(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, pcl::PointXYZRGBA& outCentroid2, bool UnitMM, bool UnitInch);

void DetermineEdgesAndCluster(std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud,
	std::string& dir,
	std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& edgesRemoved,
	bool removeEdgesAfterMeshing,
	std::vector<pcl::PointIndices>& cluster_indices);

void VoxelGridDownsample(std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud, double downsample);

void transformPc(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, Eigen::Matrix3f eig_vec);

double distance(pcl::PointXYZRGBA Old, pcl::PointXYZRGBA New);

double distance(Point Old, Point New);

Polyhedron deleteLargeTriangles(Polyhedron meshInput);

void formRemovalPlane(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr);

Polyhedron denseMesh(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr);

void samplePointsFromMesh(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, Polyhedron polyhedron, double downsampleValue, std::vector<double>& gridParameters);

bool checkForMissingPoints(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr);

double interpolateZvalueBasedOnSurroundings(double p0, double p1, double p2, double p3);

bool checkIfNoNeighborsAreMissing(double p0, double p1, double p2, double p3);

void determinePriorities(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, int iy, int ix, std::vector<double>& gridParameters, std::vector<int>& priorities);

double checkProximityPoint(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, std::vector<double>& gridParameters, int ix, int iy);

void spiralOrder(std::vector<std::vector<int>>& order, int R, int C);

void fillHolesInGrid(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, std::vector<double>& gridParameters);

void cropPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, std::vector<double>& gridParameters);

void createDenseGridPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampleValue, boost::shared_ptr< ::pcl::PolygonMesh>  meshPCL, bool stlInput);

double gaussianWeightingFunction(double cutoffWavelength, double xKI, double yLJ);

void filterShortLong(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsample, bool cropCloud, string dir, string file, float cutoffWaveLengthLow, float cutoffWaveLengthHigh);

double Sa(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr);

double Sq(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr);

std::vector<MatrixXf> CreateEvaluationLengthMatrix(double downsampling, int PointsOnVariogram, double span);

double Svr(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampling, int PointsOnVariogram, double span, std::vector<double>& SvrGauss);

void roughnessCalculation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampling, int PointsOnVariogram, double span,
	std::vector<double>& SvrGauss, std::vector<double>& SaSqSvrGauss, int RoughnessParameterSaSqSvr, bool OverWriteParameters);

double calculateSurfaceRoughnessOfPointCloud(std::string filepath);

#endif
