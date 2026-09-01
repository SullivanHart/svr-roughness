#include "cloud_viewer.h"
#include "surfinspect_native.h"
#include <tbb/parallel_for.h>
#include <filesystem>
#include <cstdlib>

using namespace tbb;

double parallelDistanceTime = 0;
double loop1Time = 0;
double loop2Time = 0;
double loop3Time = 0;
double loop4Time = 0;
double loop5Time = 0;
double loop6Time = 0;

int user_data;
// saveCloud determines if some of the points clouds in the process of the surface roughness characterization should be saved. For debugging. 
bool saveClouds = false;
PointList PointsGlobal;

//START FUNTIONS FOR MICRO EPSILON SCANNER
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudGLOBAL(new pcl::PointCloud<pcl::PointXYZRGBA>);
//For the usage with the micro epsillon sensor, because it ncessitated a exposure adjudtment. This is the goal mean grey value after the exposure adjustment.
float MeanGrey = 127;
bool GageRR = false;
// These are the values for automatically saving the roughness values for the Gage R&R study.
string Operator = "99";
string Casting = "Z";
string Run = "1";
string dir = "";
string file = "";

namespace {
bool legacyDiagnosticsEnabled() {
	const char* value = std::getenv("SURFINSPECT_LEGACY_DIAGNOSTICS");
	return value != nullptr && std::string(value) == "1";
}

std::string legacyDiagnosticPath(const std::string& dir, const std::string& file, const std::string& name) {
	const std::filesystem::path base = std::filesystem::path(dir) / (file + "_legacy_diagnostics");
	std::filesystem::create_directories(base);
	return (base / name).string();
}

void saveLegacyDiagnosticCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud,
	const std::string& dir, const std::string& file, const std::string& name) {
	if (legacyDiagnosticsEnabled()) {
		pcl::io::savePCDFileBinary(legacyDiagnosticPath(dir, file, name), *cloud);
		std::cout << "[legacy] " << name << ": " << cloud->size() << std::endl;
	}
}
}

//Determine Inverse of Matrix
Eigen::Matrix3f Inverse2(Eigen::Matrix3f matIn) {
	double determinant = 0;
	Eigen::Matrix3f matOut = Eigen::Matrix3f::Identity();
	for (int i = 0; i < 3; i++)
	{
		determinant = determinant + (matIn(0, i) * (matIn(1, (i + 1) % 3) * matIn(2, (i + 2) % 3) - matIn(1, (i + 2) % 3) * matIn(2, (i + 1) % 3)));
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
		{
			matOut(i, j) = ((matIn((j + 1) % 3, (i + 1) % 3) * matIn((j + 2) % 3, (i + 2) % 3)) -
				(matIn((j + 1) % 3, (i + 2) % 3) * matIn((j + 2) % 3, (i + 1) % 3))) / determinant;
		}
	}
	return matOut;
}

//Points have to be transformed (rotated & translated) for some operations. These transform them back. 
void TransformBack(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, Eigen::Matrix3f matIn, Eigen::Vector4f centroid) {
	Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	Eigen::Matrix3f InvsMat = Eigen::Matrix3f::Identity();
	InvsMat = Inverse2(matIn);
	// Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	transform_2.translation() << 0, 0, centroid[2];
	pcl::transformPointCloud(*inputptr, *inputptr, transform_2);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			transform_1(j, i) = InvsMat(i, j);
		}
	}
	pcl::transformPointCloud(*inputptr, *inputptr, transform_1);
}

//Performs point cloud tranfromation (rotate&translate) base on the eig_vec
Eigen::Vector4f Transformation(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, Eigen::Matrix3f eig_vec)
{
	Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	// Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			transform_1(j, i) = eig_vec(i, j);
		}
	}
	pcl::transformPointCloud(*inputptr, *inputptr, transform_1);
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*inputptr, centroid);
	transform_2.translation() << 0, 0, -centroid[2];
	pcl::transformPointCloud(*inputptr, *inputptr, transform_2);
	return centroid;
}

//Performs principal componed analysis and returns eigen vector. This can be used 
Eigen::Matrix3f PCAcalc(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr)
{
	std::cout << "PCAcalc start" << std::endl;
	pcl::PCA< pcl::PointXYZRGBA > pca;
	pca.setInputCloud(inputptr);
	Eigen::Matrix3f eig_vec = pca.getEigenVectors();
	std::cout << "PCAcalc end" << std::endl;
	return eig_vec;
}

//Passfilters that removes points outside of the bounds. 
void passfilter(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
	const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputcloudptr, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& outputcloudptr)
{
	std::cerr << "passfilter start Cloud: " << inputcloudptr->size() << std::endl;
	//for (std::size_t i = 0; i < 10; ++i)
	//  std::cerr << "    " << inputcloudptr->points[i].x << " "
	//                      << inputcloudptr->points[i].y << " "
	//                      << inputcloudptr->points[i].z << std::endl;
	pcl::PassThrough<pcl::PointXYZRGBA> pass;
	pass.setInputCloud(inputcloudptr);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(zmin, zmax);
	pass.filter(*outputcloudptr);
	pass.setInputCloud(outputcloudptr);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(xmin, xmax);
	pass.filter(*outputcloudptr);
	pass.setInputCloud(outputcloudptr);
	pass.setFilterFieldName("y");
	pass.setFilterLimits(ymin, ymax);
	pass.filter(*outputcloudptr);
	std::cerr << "passfilter end Cloud: " << inputcloudptr->size() << std::endl;
	//for (std::size_t i = 0; i < 10; ++i)
	//  std::cerr << "    " << outputcloudptr->points[i].x << " "
	//                      << outputcloudptr->points[i].y << " "
	//                      << outputcloudptr->points[i].z << std::endl;
}

//used for downsampling to an evenly spaced grid (useful for gaussian filter application)
void passfilter2DGrid(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampleValue) {
	std::cerr << "passfilter2DGrid start Cloud: " << inputptr->size() << std::endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr organizedCloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr testPC(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointXYZRGBA minPt, maxPt;
	//pcl::PointXYZRGBA p;
	//Setting up corner points of grid
	pcl::getMinMax3D(*inputptr, minPt, maxPt);
	int stepsX = ceil((maxPt.x - minPt.x) / downsampleValue) + 1;
	int stepsY = ceil((maxPt.y - minPt.y) / downsampleValue) + 1;
	double xVal = minPt.x;
	double yVal = minPt.y;
	Eigen::Vector4f centroid;
	organizedCloud->width = stepsX;
	organizedCloud->height = stepsY;
	organizedCloud->is_dense = false;
	organizedCloud->points.resize(organizedCloud->height * organizedCloud->width);
	pcl::PointXYZRGBA searchPoint;
	// K nearest neighbor search
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(inputptr);
	int K = 3;
	std::vector<int> pointIdxKNNSearch(K);
	std::vector<float> pointKNNSquaredDistance(K);
	for (int ix = 0; ix < stepsX; ix++) {
		for (int iy = 0; iy < stepsY; iy++) {
			passfilter(xVal + ix * downsampleValue, xVal + ix * downsampleValue + downsampleValue, yVal + iy * downsampleValue, yVal + iy * downsampleValue + downsampleValue, -10000, 10000, inputptr, testPC);
			int PointsUsedForCenter = pcl::compute3DCentroid(*testPC, centroid);
			if (PointsUsedForCenter > 0) {
				double x = (double)(xVal + ix * downsampleValue + downsampleValue / 2);
				double y = (double)(yVal + iy * downsampleValue + downsampleValue / 2);
				//populating grid values
				organizedCloud->at(ix, iy).x = x;
				organizedCloud->at(ix, iy).y = y;
				organizedCloud->at(ix, iy).z = centroid[2];
			}
			else {
				double x = (double)(xVal + ix * downsampleValue + downsampleValue / 2);
				double y = (double)(yVal + iy * downsampleValue + downsampleValue / 2);
				searchPoint.x = x;
				searchPoint.y = y;
				searchPoint.z = 0;
				double z = 0;
				if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
				{
					for (std::size_t i = 0; i < pointIdxKNNSearch.size(); ++i) {
						z += (*inputptr)[pointIdxKNNSearch[i]].z;
					}
				}
				organizedCloud->at(ix, iy).x = x;
				organizedCloud->at(ix, iy).y = y;
				organizedCloud->at(ix, iy).z = z / K;
			}
		}
	}
	pcl::copyPointCloud(*organizedCloud, *inputptr);
	std::cerr << "passfilter2DGrid end Cloud: " << inputptr->size() << std::endl;
}

// simple downsampling to a single layer of points. Points are not distributed on a perfect x/y grid. x/y location based on mean of voxel.
void passfilter2D(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampleValue) {
	std::cerr << "passfilter2D start Cloud: " << inputptr->size() << std::endl;
	pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
	vg.setInputCloud(inputptr);
	vg.setLeafSize(downsampleValue, downsampleValue, 10);
	vg.filter(*inputptr);
	std::cerr << "passfilter2D end Cloud: " << inputptr->size() << std::endl;
}

//Downsampling bsed on a grid. To achieve this the point cloud first has to be aligned with the x/y axis. Then downsalmpling. Finally transforming back. 
void GridFilter(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampleValue) {
	std::cerr << "GridFilter start Cloud: " << inputptr->size() << std::endl;
	Eigen::Matrix3f eigen_vector[1];
	std::vector<Eigen::Vector4f> centroid_vec;
	eigen_vector[0] = PCAcalc(inputptr);
	centroid_vec.push_back(Transformation(inputptr, eigen_vector[0]));
	passfilter2D(inputptr, downsampleValue);
	TransformBack(inputptr, eigen_vector[0], centroid_vec[0]);
	std::cerr << "GridFilter end Cloud: " << inputptr->size() << std::endl;
}

//Disatance calculation between surface and points. Using a tree as infut for efficient calculation
void  ParallelDistance(std::vector<double>& output, PointList input, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointsOnSurface, Tree* inputTree,
	size_t n/*, CGAL::Side_of_triangle_mesh<Polyhedron, Kernel> inside*/) {
	std::cerr << "ParallelDistance start" << std::endl;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//for (int i = 0; i < n; i++) {
	//https://doc.cgal.org/latest/AABB_tree/index.html
	parallel_for(size_t(0), n, [&](size_t i) {
		output[i] = sqrt(inputTree->squared_distance(input[i].first));
		Point_and_primitive_id pp = inputTree->closest_point_and_primitive(input[i].first);
		Polyhedron::Face_handle f = pp.second.first; // closest primitive id
		CGAL::Vector_3<Kernel> v1 = input[i].first - pp.first;
		double angle = std::acos((v1 * input[i].second / CGAL::sqrt(v1 * v1) / CGAL::sqrt(input[i].second * input[i].second))) * (180 / 3.14159265358979323846);
		// handeling points on the border
		if (f->facet_begin()->is_border_edge()) {
			if ((angle > 20 && angle < 160) || (angle > 200 && angle < 340)) {
				output[i] = 1;
			}
		}
		if (f->facet_begin()->next()->is_border_edge()) {
			if ((angle > 20 && angle < 160) || (angle > 200 && angle < 340)) {
				output[i] = 1;
			}
		}
		if (f->facet_begin()->next()->next()->is_border_edge()) {
			if ((angle > 20 && angle < 160) || (angle > 200 && angle < 340)) {
				output[i] = 1;
			}
		}
		pcl::PointXYZRGBA point;
		point.x = pp.first.x();
		point.y = pp.first.y();
		point.z = pp.first.z();
		pointsOnSurface->points[i] = point;
		if (CGAL::angle(v1, input[i].second) == CGAL::OBTUSE) {
			output[i] = -output[i];
		}
		//https://doc.cgal.org/latest/Polygon_mesh_processing/index.html
		});
	//}
	std::chrono::steady_clock::time_point endtime = std::chrono::steady_clock::now();
	parallelDistanceTime += std::chrono::duration_cast<std::chrono::microseconds>(endtime - begin).count();
	std::cerr << "ParallelDistance end Time: " << parallelDistanceTime << std::endl;
}

//This function performs the majority of the work to determine the variogram roughness of the point cloud. Parralel computation.                                                                                                                                
double calcVariogramFromPCParallel(std::vector<double>& SvrPoint2,
			const PointCloud<PointXYZRGBA>::Ptr& cloud,
			std::vector<double> dist,
			std::vector<double>& varVec,
			std::vector<double>& sumVec,
			std::vector<double>& CtrVec,
			int PointsOnVariogram,
			double span) {
	std::cout << "calcVariogramFromPCParallel start" << endl;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	const int NmbrOfPtInPC = cloud->size();// points2.size();
	size_t size = cloud->size();// points2.size();
	std::vector<std::vector<double> > sumVecParallel(size, std::vector<double>(PointsOnVariogram + 1));
	std::vector<std::vector<double> > CtrVecParallel(size, std::vector<double>(PointsOnVariogram + 1));
	std::vector<double> SvrPoint(cloud->size(), 0);
	//Create Kd tree for fast distance calculations                                               
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(cloud);
	//Identify points withing range and z value difference.
	parallel_for(size_t(0), size, [&](size_t i) {
		//for (int i = 0; i < size; i++) {
		pcl::PointXYZRGBA searchPoint = cloud->points[i];
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;
		float radius = PointsOnVariogram * span;
		//Find points that are within the radius from the search point.                                                                 
		kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
		for (int j = 1; j < pointIdxRadiusSearch.size(); j++) {
			int k = floor(sqrt(pointRadiusSquaredDistance[j]) / span);      //IS THIS NEEDED CAN THIS BE REMOVED? No sirts the values in the correct variogram distance
			if (k > -1) {
				// Determine the squared distance between points and surface.                                                                  
				sumVecParallel[i][k] += pow(1000 * (dist[i] - dist[pointIdxRadiusSearch[j]]), 2);
				CtrVecParallel[i][k] += 1;
			}
		}
		int ctr = 0;
		for (int m = 0; m < PointsOnVariogram; m++) {
			if (CtrVecParallel[i][m] > 0) {
				//This is part of the variogram equation                                       
				SvrPoint[i] += sqrt((1 / (2 * CtrVecParallel[i][m])) * sumVecParallel[i][m]);
				ctr++;
			}
		}
		//Roughness value of point i                          
		SvrPoint[i] = SvrPoint[i] / ctr;
		});
	//}
	double /*Svr1=0, Svr2=0,*/ Svr3 = 0;
	double CtrAll = 0;
	//combine sums and counters to determine variogram roughness value.
	for (int i = 0; i < PointsOnVariogram; i++) {
		//for (int j = 0; j < size; j++) {
		sumVec[i] = 0;
		CtrVec[i] = 0;
		varVec[i] = 0;
		for (int j = 0; j < cloud->size(); j++) {
			sumVec[i] += sumVecParallel[j][i];// accumulate(sumVecParallel[i].begin(), sumVecParallel[i].end(), 0);
			CtrVec[i] += CtrVecParallel[j][i];// accumulate(CtrVecParallel[i].begin(), CtrVecParallel[i].end(), 0);
		}
		if (CtrVec[i] > 0) {
			varVec[i] = sqrt((1 / (2 * CtrVec[i])) * sumVec[i]);
			CtrAll += CtrVec[i];
			cout << std::setprecision(6) << "var[" << i << "] = " << varVec[i] << endl;
			Svr3 += varVec[i];
		}
		else {
			cout << "Warning: no point pairs were found for: " << i << endl;
		}
	}
	//Svr1 = Svr1 / PointsOnVariogram;
	Svr3 = Svr3 / PointsOnVariogram;
	SvrPoint2 = SvrPoint;
	return Svr3;
	std::cout << "calcVariogramFromPCParallel end time: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count() << "[s]" << std::endl;
}

// local roughness calculation: advantage no meshing required, but not as accurate so far.
std::vector<double> calcVariogramFromPCParallel_local(PointList points2,
			std::vector<double>& varVec,
			std::vector<double>& sumVec,
			std::vector<double>& CtrVec,
			int PointsOnVariogram,
			double span,
			bool exactDistance) {
	std::cout << "calcVariogramFromPCParallel_local start" << endl;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	for (int i = 0; i < points2.size(); i++) {
		pcl::PointXYZRGBA p;
		p.x = points2[i].first.x();
		p.y = points2[i].first.y();
		p.z = points2[i].first.z();
		cloud->push_back(p);
	}
	const int NmbrOfPtInPC = cloud->size();
	size_t size = cloud->size();
	std::vector<std::vector<double> > sumVecParallel(size, std::vector<double>(PointsOnVariogram + 1));
	std::vector<std::vector<double> > CtrVecParallel(size, std::vector<double>(PointsOnVariogram + 1));
	std::vector<double> SvrPoint(cloud->size(), 0);
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(cloud);
	parallel_for(size_t(0), size, [&](size_t i) {
		//for (int i = 0; i < size; i++) {
		if (i % 1000 == 0) {
			cout << "i: " << i << endl;
		}
		pcl::PointXYZRGBA searchPoint = cloud->points[i];
		PointList pointsLocal;
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;
		float radius = PointsOnVariogram * span;
		kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
		Eigen::Matrix3f eigen_vector[1];
		std::vector<Eigen::Vector4f> centroid_vec;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr inputptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr inputptr2(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::copyPointCloud(*cloud, *inputptr2);
		for (int m = 0; m < pointIdxRadiusSearch.size(); m++) {
			pcl::PointXYZRGBA p;
			p = inputptr2->points[pointIdxRadiusSearch[m]];
			pointsLocal.push_back(points2[pointIdxRadiusSearch[m]]);
			inputptr->push_back(p);
		}
		eigen_vector[0] = PCAcalc(inputptr);
		centroid_vec.push_back(Transformation(inputptr, eigen_vector[0]));
		int sign = 1;
		for (int j = 1; j < inputptr->size(); j++) {
			int k = ceil(sqrt(pointRadiusSquaredDistance[j]) / span) - 1;
			sumVecParallel[i][k] += pow(1000 * (searchPoint.z - inputptr->points[j].x), 2);
			CtrVecParallel[i][k] += 1;
		}
		int ctr = 0;
		for (int m = 0; m < PointsOnVariogram; m++) {
			if (CtrVecParallel[i][m] > 0) {
				SvrPoint[i] += sqrt((1 / (2 * CtrVecParallel[i][m])) * sumVecParallel[i][m]);
				ctr++;
			}
		}
		SvrPoint[i] = SvrPoint[i] / ctr;
		});
	double Svr1 = 0, Svr2 = 0, Svr3 = 0;
	//combine sums and counters
	for (int i = 0; i < PointsOnVariogram; i++) {
		sumVec[i] = 0;
		CtrVec[i] = 0;
		varVec[i] = 0;
		for (int j = 0; j < cloud->size(); j++) {
			sumVec[i] += sumVecParallel[j][i];// accumulate(sumVecParallel[i].begin(), sumVecParallel[i].end(), 0);
			CtrVec[i] += CtrVecParallel[j][i];// accumulate(CtrVecParallel[i].begin(), CtrVecParallel[i].end(), 0);

		}
		if (CtrVec[i] > 0) {
			varVec[i] = sqrt((1 / (2 * CtrVec[i])) * sumVec[i]);
			cout << std::setprecision(6) << "var[" << i << "] = " << varVec[i] << endl;
			Svr3 += varVec[i];
		}
		else {
			cout << "Warning: no point pairs were found for: " << i << endl;
		}
	}
	Svr1 = Svr1 / PointsOnVariogram;
	Svr3 = Svr3 / PointsOnVariogram;
	for (int i = 0; i < size; i++) {
		if (isnan(SvrPoint[i]) == 0) {
			Svr2 += SvrPoint[i];
		}
		else {
			cout << "isnan: " << i << endl;
			pcl::PointXYZRGBA searchPoint = cloud->points[i];
			std::vector<int> pointIdxRadiusSearch;
			std::vector<float> pointRadiusSquaredDistance;
			float radius = PointsOnVariogram * span;
			kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
			cout << "pointIdxRadiusSearch.size(): " << pointIdxRadiusSearch.size() << endl;
			for (int m = 0; m < PointsOnVariogram; m++) {
				cout << "sumVecParallel[i][m]: " << sumVecParallel[i][m] << endl;
				cout << "CtrVecParallel[i][m]: " << CtrVecParallel[i][m] << endl;
			}
		}
	}
	Svr2 = Svr2 / int(size);
	cout << std::setprecision(15) << "Svr1: " << Svr1 << endl;
	cout << std::setprecision(15) << "Svr2: " << Svr2 << endl;
	cout << std::setprecision(15) << "Svr3: " << Svr3 << endl;
	std::cout << "calcVariogramFromPCParallel_local end time: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count() << "[s]" << std::endl;
	return SvrPoint;
}

// Downsample a CGAL point Cloud
void downsampleCGAL(PointList& points, double cell_size_grid_simplify_point_set) {
	std::vector<std::size_t> indices(points.size());
	std::vector<Point> tmp_points(points.size());

	for (std::size_t i = 0; i < points.size(); ++i) {
		indices[i] = i;
		tmp_points[i] = points[indices[i]].first;
	}

	// Downsample CGAL point cloud
	std::vector<std::size_t>::iterator end;
	end = CGAL::grid_simplify_point_set(
		indices,
		cell_size_grid_simplify_point_set,
		CGAL::parameters::point_map(CGAL::make_property_map(tmp_points))
	);

	std::size_t k = end - indices.begin();
	{
		std::vector<Point_with_normal> tmp_p(k);
		std::vector<Kernel::Vector_3> tmp_normals(k); // Use Vector from your typedefs

		for (std::size_t i = 0; i < k; ++i) {
			tmp_p[i].first = points[indices[i]].first;
		}

		points.swap(tmp_p);
	}
}

//Check normal orientation of point cloud. If the normals are not aligned correctly it is difficult to determine 
//if two points are on the same side of the complex mesh or opposing sides
void checkNormalOrientation(PointList& points, pcl::PointXYZRGBA Centroid) {
	cout << "checkNormalOrientation start" << endl;
	PointCloud<PointNormal>::Ptr cloud_smoothed_normals(new PointCloud<PointNormal>());
	for (int i = 0; i < points.size(); i++) {
		pcl::PointNormal p;
		p.x = points[i].first.x();
		p.y = points[i].first.y();
		p.z = points[i].first.z();
		p.normal_x = points[i].second.x();
		p.normal_y = points[i].second.y();
		p.normal_z = points[i].second.z();
		cloud_smoothed_normals->push_back(p);
	}
	pcl::PointNormal CentroidNormal;
	pcl::computeCentroid(*cloud_smoothed_normals, CentroidNormal);
	Kernel::Vector_3 NormalCentroid(CentroidNormal.normal_x, CentroidNormal.normal_y, CentroidNormal.normal_z); // = points[1].second;
	Kernel::Vector_3 LocaltoGlobalCentroidVector(CentroidNormal.x - Centroid.x, CentroidNormal.y - Centroid.y, CentroidNormal.z - Centroid.z);
	if (CGAL::angle(NormalCentroid, LocaltoGlobalCentroidVector) == CGAL::OBTUSE) {
		for (int i = 0; i < points.size(); i++) {
			Kernel::Vector_3 VectorBuffer(-points[i].second.x(), -points[i].second.y(), -points[i].second.z());
			points[i].second = VectorBuffer;
		}
		cout << "checkNormalOrientation: Normals were aligned." << endl;
	}
	else {
		cout << "checkNormalOrientation: Normals are aligned correctly. NO action needed." << endl;
	}
	cout << "checkNormalOrientation end" << endl;
}

// This determines the Normals of a CGAL Point Cloud
void estimateNormalsCGAL(PointList& points) {
	cout << "estimateNormalsCGAL start" << endl;
	int nb_neighbors = 18; // K-nearest neighbors = 3 rings
	CGAL::pca_estimate_normals<Concurrency_tag>(points, nb_neighbors, CGAL::parameters::point_map(Point_map()).normal_map(Normal_map()));
	PointList points2 = points;
	// Orients normals.
	nb_neighbors = 18;
	// Note: mst_orient_normals() requires a range of points         // as well as property maps to access each point's position and normal. 
	//https://docs.huihoo.com/cgal/3.7/cgal-manual/Point_set_processing_3_ref/Function_mst_orient_normals.html#Cross_link_anchor_1596
	std::vector<Point_with_normal>::iterator unoriented_points_begin = CGAL::mst_orient_normals(points, nb_neighbors, CGAL::parameters::point_map(Point_map()).normal_map(Normal_map()));
	// Optional: delete points with an unoriented normal// if you plan to call a reconstruction algorithm that expects oriented normals.
	points.erase(unoriented_points_begin, points.end());
	cout << "estimateNormalsCGAL end: points left after orienting point cloud (small is bad): " << points.size() << endl;
}

// Cretate a mesh from bezier (disabled, experimental)
void createMeshBezier(PointList& points, int loopCounter, string dir, string file, double target_edge_length) {
	std::cout << "createMeshBezier (disabled)" << std::endl;
}

// Part of the meshing operation. First create a very dense mesh from a dense point cloud
Polyhedron createDenseMesh(Polyhedron output_mesh_buffer, PointList& points, int loopCounter, string dir, string file) {
	cout << "createDenseMesh start" << endl;
	Point_set PointsSet;
	Mesh2 mesh;
	PointsSet.add_normal_map();
	for (int k = 0; k < points.size(); k++) {
		Point_set::iterator new_item = PointsSet.insert(points[k].first, points[k].second);
	}
	typedef std::array<std::size_t, 3> Facet; // Triple of indices
	std::vector<Facet> facets;
	// this does all the work:
	CGAL::advancing_front_surface_reconstruction(PointsSet.points().begin(), PointsSet.points().end(), std::back_inserter(facets));
	std::cout << facets.size() << " facet(s) generated by reconstruction." << std::endl;
	// copy points for random access
	std::vector<Point> vertices;
	vertices.reserve(PointsSet.points().size());
	std::copy(PointsSet.points().begin(), PointsSet.points().end(), std::back_inserter(vertices));
	CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(vertices, facets, output_mesh_buffer);
	cout << "createDenseMesh end" << endl;
	return output_mesh_buffer;
}

//Nexty remesh to make a sparse but uniform mesh.
Polyhedron remesh(Polyhedron& mesh2, int loopCounter, double target_edge_length, string dir, string file, bool secondRun) {
	cout << "remesh start" << endl;
	std::stringstream ss;
	if (secondRun == false) {
		ss << dir << file << "AdvancingMesh" << loopCounter << ".off";
	}
	else {
		ss << dir << file << "Remeshed" << loopCounter << ".off";
	}
	std::ifstream input(ss.str().c_str());
	Mesh2 mesh;
	CGAL::copy_face_graph(mesh2, mesh);
	mesh2.clear();
	unsigned int nb_iter = 10;
	std::vector<edge_descriptor> border;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
	CGAL::Polygon_mesh_processing::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
	CGAL::Polygon_mesh_processing::split_long_edges(border, target_edge_length, mesh);
	std::cout << "remesh    Start remeshing of " << "advancing_front.off" << " (" << num_faces(mesh) << " faces)..." << std::endl;
	CGAL::Polygon_mesh_processing::isotropic_remeshing(faces(mesh), target_edge_length, mesh, CGAL::Polygon_mesh_processing::parameters::number_of_iterations(nb_iter)
		.protect_constraints(true)//i.e. protect border
	);
	CGAL::copy_face_graph(mesh, mesh2);
	cout << "remesh end" << " (" << num_faces(mesh) << " faces)..." << endl;
	return mesh2;
}

//Color Point Cloud based on height values stored in dist2
void colorPcBasedOnDist(std::vector<double>& dist2, PointCloud<PointXYZRGBA>::Ptr& cloud) {
	cout << "colorDist start" << endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBA(new pcl::PointCloud<pcl::PointXYZRGBA>);
	double maxD2 = *max_element(dist2.begin(), dist2.end());
	double minD2 = *min_element(dist2.begin(), dist2.end());
	double maxValD2 = abs(minD2);
	if (abs(maxD2) > abs(minD2)) {
		maxValD2 = abs(maxD2);
	}
	pcl::copyPointCloud(*cloud, *cloudRGBA);
	parallel_for(size_t(0), cloudRGBA->size(), [&](size_t i) {
		int colorVal = abs(((dist2[i]) / (maxValD2)) * 255);
		if (colorVal > 255) {
			colorVal = 255;
		}
		int r, g, b;
		if (dist2[i] > 0) {
			r = (colorVal);
			g = 255 - r;
			b = 0;
		}
		else {
			b = (colorVal);
			g = 255 - b;
			r = 0;
		}
		cloudRGBA->points[i].r = r;
		cloudRGBA->points[i].g = g;
		cloudRGBA->points[i].b = b;
		});
	pcl::copyPointCloud(*cloudRGBA, *cloud);
	cout << "colorDist end" << endl;
}

//This function colors the point cloud based on the distance from point cloud to mesh and saves them
void colorPcBasedOnDist(std::vector<double>& dist2, PointCloud<PointXYZRGBA>::Ptr& cloud, string dirFile, int number) {
	cout << "colorDist start" << endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBA(new pcl::PointCloud<pcl::PointXYZRGBA>);
	double maxD2 = 0.001;//*max_element(dist2.begin(), dist2.end());
	double minD2 = 0.001;//*min_element(dist2.begin(), dist2.end());
	double maxValD2 = abs(minD2);
	if (abs(maxD2) > abs(minD2)) {
		maxValD2 = abs(maxD2);
	}
	pcl::copyPointCloud(*cloud, *cloudRGBA);
	parallel_for(size_t(0), cloudRGBA->size(), [&](size_t i) {
		int colorVal = abs(((dist2[i]) / (maxValD2)) * 255);
		if (colorVal > 255) {
			colorVal = 255;
		}
		int r, g, b;
		if (dist2[i] > 0) {
			r = (colorVal);
			g = 255 - r;
			b = 0;
		}
		else {
			b = (colorVal);
			g = 255 - b;
			r = 0;
		}
		cloudRGBA->points[i].r = r;
		cloudRGBA->points[i].g = g;
		cloudRGBA->points[i].b = b;
		});
	cout << "colorDist end" << endl;
}

//This changes the colors of the abnormalities in a point cloud
void colorPcBasedOnAbnormalities(std::vector<double>& SvrPointCombined, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudRGBA, double SvrFinal, std::vector<double>& dist2Combined,
	int abnormalitySelection) {
	cout << "colorPcAbnormalities start" << endl;
	cout << "colorPcAbnormalities:  abnormalitySelection: " << abnormalitySelection << endl;
	parallel_for(size_t(0), cloudRGBA->size(), [&](size_t i) {
		int r, g, b;
		if (SvrPointCombined[i] > 2 * SvrFinal) {
			//depending on the selection only depressions, only hills or both are considered abnormalities and colored  
			if (abnormalitySelection == 1 && dist2Combined[i] < 0) {
				r = 0;
				g = 0;
				b = 255;
			}
			else if (abnormalitySelection == 2 && dist2Combined[i] > 0) {
				r = 255;
				g = 0;
				b = 0;
			}
			else if (abnormalitySelection == 0 && dist2Combined[i] < 0) {
				r = 0;
				g = 0;
				b = 255;
			}
			else if (abnormalitySelection == 0 && dist2Combined[i] > 0) {
				r = 255;
				g = 0;
				b = 0;
			}
			else {
				r = 0;
				g = 255;
				b = 0;
			}
		}
		else {
			r = 0;
			g = 255;
			b = 0;
		}
		cloudRGBA->points[i].r = r;
		cloudRGBA->points[i].g = g;
		cloudRGBA->points[i].b = b;
		});
	cout << "colorPcAbnormalities end" << endl;
}

// Colors the point cloud based on the associated variogram values. This is very useful or analyzing large castings where a combined roughness result is somewhat meaningless. 
//go from white(0) to blue(0.04) to green (0.055) to yellow(0.072) to red (0.13) to black ( 0.2)
void colorPcBasedOnLocalVariogram(std::vector<double>& SvrPoint, PointCloud<PointXYZRGBA>::Ptr& cloudRGBA) {
	cout << "colorPcVariogram start" << endl;
	parallel_for(size_t(0), cloudRGBA->size(), [&](size_t i) {
		int r, g, b;
		if (SvrPoint[i] < 0.03) {
			//white(255,255,255) to blue (0,0,255)
			r = -6375 * SvrPoint[i] + 255; // y=mx+b
			g = -6375 * SvrPoint[i] + 255;
			b = 255;
		}
		else if (SvrPoint[i] >= 0.03 && SvrPoint[i] < 0.048) {
			//blue(0,0,255) to green(0,255,0)
			r = 0;
			g = 17000 * SvrPoint[i] + -680;
			b = -17000 * SvrPoint[i] + 935;
		}
		else if (SvrPoint[i] >= 0.048 && SvrPoint[i] < 0.067) {
			//green(0,255,0) to yellow(255,255,0)
			r = 15000 * SvrPoint[i] + -825;
			g = 255;
			b = 0;
		}
		else if (SvrPoint[i] >= 0.067 && SvrPoint[i] < 0.14) {
			//yellow(255,255,0) to red(255,0,0)
			r = 255;
			g = -4396.5517241379 * SvrPoint[i] + 571.55172413793;
			b = 0;
		}
		else if (SvrPoint[i] >= 0.14 && SvrPoint[i] < 0.2) {
			//red(255,0,0) to black(0,0,0) 
			r = -3642.8571428571 * SvrPoint[i] + 728.57142857143;
			g = 0;
			b = 0;
		}
		cloudRGBA->points[i].r = r;
		cloudRGBA->points[i].g = g;
		cloudRGBA->points[i].b = b;
		});
	cout << "colorPcVariogram end" << endl;
}

//prepping point cloud for variogram roughness calculation
void Prep(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer, const pcl::PointXYZRGBA& outCentroid2, PointList& points, PointList& points2, int j,
	std::string& dir, std::string& file, bool statFilter, bool GridFilterBool, bool removeEdgesAfterMeshing, double downsample, double cell_size_grid_simplify_point_set)
{
	cout << "#######################################################################" << endl;
	cout << "clusterPrepNoCluster start  j: " << j << endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::copyPointCloud(*cloud_buffer, *cloud_cluster);

	//Statistical filter if choosen
	if (statFilter) {
		StatOutlierRemoval(cloud_cluster);
	}
	//Gridfilter if choosen
	if (GridFilterBool == true && removeEdgesAfterMeshing == false) {
		GridFilter(cloud_cluster, downsample);
		StatOutlierRemoval(cloud_cluster);
	}
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudTemp(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::copyPointCloud(*cloud_cluster, *cloud_buffer);
	for (int i = 0; i < cloud_cluster->size(); i++) {
		Point pt(cloud_cluster->points[i].x, cloud_cluster->points[i].y, cloud_cluster->points[i].z);
		Point_with_normal pn;
		pn.first = pt;
		points2.push_back(pn);
	}
	//Downsample point cloud
	pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
	vg.setInputCloud(cloud_cluster);
	vg.setLeafSize(cell_size_grid_simplify_point_set, cell_size_grid_simplify_point_set, cell_size_grid_simplify_point_set);
	vg.filter(*cloud_cluster);
	vg.setInputCloud(cloud_cluster);
	vg.setLeafSize(cell_size_grid_simplify_point_set, cell_size_grid_simplify_point_set, cell_size_grid_simplify_point_set);
	vg.filter(*cloud_cluster);
	//if (saveClouds) {
	//	pcl::io::savePCDFileASCII(dir + file + "Dowsmpl.pcd", *cloud_cluster);
	//}
	for (int i = 0; i < cloud_cluster->size(); i++) {
		Point pt(cloud_cluster->points[i].x, cloud_cluster->points[i].y, cloud_cluster->points[i].z);
		Point_with_normal pn;
		pn.first = pt;
		points.push_back(pn);
	}
	estimateNormalsCGAL(points);
	estimateNormalsCGAL(points2);
	checkNormalOrientation(points2, outCentroid2);
	cout << "clusterPrepNoCluster end" << j << endl;
	cout << "#######################################################################" << endl;
}

//prepping point cloud for variogram roughness calculation
void clusterPrep(int j, std::vector<pcl::PointIndices>::const_iterator& it, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& edgesRemoved, /*std::stringstream& ss,*/ std::string& dir,
	std::string& file, bool statFilter, bool GridFilterBool, bool removeEdgesAfterMeshing, double downsample, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer,
	PointList& points2, double cell_size_grid_simplify_point_set, PointList& points, const pcl::PointXYZRGBA& outCentroid2)
{
	cout << "#######################################################################" << endl;
	cout << "clusterPrep start  j: " << j << endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr buffer(new pcl::PointCloud<pcl::PointXYZRGBA>);
	std::stringstream ss;
	for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		cloud_cluster->push_back((*edgesRemoved)[*pit]); //*
	cloud_cluster->width = cloud_cluster->size();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;
	//if (saveClouds) {
	//	ss << dir << file << "cloud_cluster_" << j << ".pcd";
	//	pcl::io::savePCDFileASCII(ss.str(), *cloud_cluster);
	//}
	//remove statistical outliers if selected
	if (statFilter) {
		StatOutlierRemoval(cloud_cluster);
	}
	//gridfiltering can also inprove roughness calculation if the scanner produces pretty noisy data
	if (GridFilterBool == true && removeEdgesAfterMeshing == false) {
		GridFilter(cloud_cluster, downsample);

		StatOutlierRemoval(cloud_cluster);
	}
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudTemp(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::copyPointCloud(*cloud_cluster, *buffer);

	//edges can be removed after meshing, but in gerneral removing edges first is better since edge point decrease the accuracy of the underlying geometry meshing
	if (removeEdgesAfterMeshing) {
		pcl::PointIndices::Ptr inliers2(new pcl::PointIndices());

		//Indices are needed so the points can be identified and deleted.
		pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
		for (int i = 0; i < cloud_cluster->size(); i++)
		{
			if (cloud_cluster->points[i].r > 239) // e.g. remove all pts below zAvg
			{
				inliers2->indices.push_back(i);
			}
		}
		extract.setInputCloud(cloud_cluster);
		extract.setIndices(inliers2);
		extract.setNegative(false);
		extract.filter(*cloud_cluster);
	}
	pcl::copyPointCloud(*cloud_cluster, *cloud_buffer);
	for (int i = 0; i < cloud_buffer->size(); i++) {
		Point pt(cloud_buffer->points[i].x, cloud_buffer->points[i].y, cloud_buffer->points[i].z);
		Point_with_normal pn;
		pn.first = pt;
		points2.push_back(pn);
	}
	//Downsample cloud
	pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
	vg.setInputCloud(buffer);
	vg.setLeafSize(cell_size_grid_simplify_point_set, cell_size_grid_simplify_point_set, cell_size_grid_simplify_point_set);
	vg.filter(*buffer);
	vg.setInputCloud(buffer);
	vg.setLeafSize(cell_size_grid_simplify_point_set, cell_size_grid_simplify_point_set, cell_size_grid_simplify_point_set);
	vg.filter(*buffer);
	//if (saveClouds) {
	//	pcl::io::savePCDFileASCII(dir + file + "Dowsmpl.pcd", *buffer);
	//}
	for (int i = 0; i < buffer->size(); i++) {
		Point pt(buffer->points[i].x, buffer->points[i].y, buffer->points[i].z);
		Point_with_normal pn;
		pn.first = pt;
		points.push_back(pn);
	}
	//determine normals.
	estimateNormalsCGAL(points);
	estimateNormalsCGAL(points2);
	checkNormalOrientation(points2, outCentroid2);
	cout << "clusterPrep end" << j << endl;
	cout << "#######################################################################" << endl;
}

//Output cluster values
void outputClusterCommandLine(std::vector<pcl::PointIndices>& cluster_indices, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr edgesRemoved)
{
	cout << "outputClusterCommandLine start" << endl;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGBA>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			cloud_cluster->push_back((*edgesRemoved)[*pit]); //*
		cloud_cluster->width = cloud_cluster->size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
		std::cout << "outputClusterCommandLine PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
	}
	cout << "outputClusterCommandLine end" << endl;
}

// Function to create Gaussian filter
MatrixXf crateGaussFilterKernel(MatrixXf GKernel, double cutoffWaveLength, int KernelSize, bool high)
{
	cout << "crateGaussFilterKernel start" << endl;
	// sum is for normalization
	double sum = 0.0;
	double alpha = 0.4697;
	double pi = 3.14159265358979323846;
	int max = (KernelSize - 1) / 2;
	int min = -max;
	float factor = 0.0002;
	// generating 5x5 kernel
	for (int x = min; x <= max; x++) {
		for (int y = min; y <= max; y++) {
			if (high) {
				GKernel(x + max, y + max) = (1 / (pow(alpha, 2) * pow(cutoffWaveLength, 2))) * exp((-pi / pow(alpha, 2)) * ((pow((x * factor), 2) + pow((y * factor), 2)) / pow(cutoffWaveLength, 2)));
			}
			else {
				GKernel(x + max, y + max) = (1 / (pow(alpha, 2) * pow(cutoffWaveLength, 2))) * exp((-pi / pow(alpha, 2)) * ((pow((x * factor), 2) + pow((y * factor), 2)) / pow(cutoffWaveLength, 2)));
			}
			sum += GKernel(x + max, y + max);
		}
	}
	// normalising the Kernel
	for (int i = 0; i < KernelSize; ++i)
		for (int j = 0; j < KernelSize; ++j)
			GKernel(i, j) /= sum;
	cout << "crateGaussFilterKernel end" << endl;
	return GKernel;
}

//Applies the gaussian filter to a point cloud. Thus the z values of the point cloud are changed
void applyFilterKernelToPc(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, MatrixXf GKernelLow, MatrixXf GKernelHigh, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudXYZ, int KernelSize) {
	cout << "applyFilterKernelToPc start" << endl;
	int maxX = inputptr->width;
	int maxY = inputptr->height;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr organizedCloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
	int max = (KernelSize - 1) / 2;
	int min = -max;
	for (int i = max; i < maxX - max; ++i) {
		for (int j = max; j < maxY - max; ++j) {
			double w = 0;
			for (int x = min; x <= max; x++) {
				for (int y = min; y <= max; y++) {
					w += GKernelLow(x + max, y + max) * (1 - GKernelHigh(x + max, y + max)) * inputptr->at(i + x, j + y).z;
				}
			}
			pcl::PointXYZRGBA p;
			p.x = inputptr->at(i, j).x;
			p.y = inputptr->at(i, j).y;
			p.z = w;
			organizedCloud->push_back(p);
			if (i > 20 && i < 23 && j > 20 && j < 23) {
				cout << "w: " << w << endl;
				cout << "z: " << inputptr->at(i, j).z << endl;
			}
		}
	}
	pcl::PointXYZRGBA minPt2, maxPt2;
	pcl::getMinMax3D(*organizedCloud, minPt2, maxPt2);
	passfilter(minPt2.x, maxPt2.x, minPt2.y, maxPt2.y, -10, 10, cloudXYZ, cloudXYZ);
	pcl::copyPointCloud(*organizedCloud, *inputptr);
	cout << "applyFilterKernelToPc end" << endl;
}

void applyGaussianFilterToCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, MatrixXf GKernel, int KernelSize, bool HighPass, bool cropCloud, string dir, string file) {
	cout << "applyFilterCloud Start:  HighPass: " << HighPass << " GKernel: " << GKernel.size() << " GKernel.rows(): " << GKernel.rows()
		<< " GKernel.size()/GKernel.rows(): " << GKernel.size() / GKernel.rows() << " GKernel.cols(): " << GKernel.cols() << endl;
	if (HighPass) {
		//pcl::io::savePCDFileASCII("C:\\Users\\dschimpf\\Box\\Daniel Schimpf\\RE\\Surface Project\\PCL_Project\\PC_svr2 - Desktop\\build\\PCD\\ApplyFilterCloudStartLarge.pcd", *inputptr);
	}
	else {
		//pcl::io::savePCDFileASCII("C:\\Users\\dschimpf\\Box\\Daniel Schimpf\\RE\\Surface Project\\PCL_Project\\PC_svr2 - Desktop\\build\\PCD\\ApplyFilterCloudStartSmall.pcd", *inputptr);
	}
	int maxX = inputptr->width;
	int maxY = inputptr->height;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr organizedCloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr organizedCloud2(new pcl::PointCloud<pcl::PointXYZRGBA>());
	int max = (KernelSize - 1) / 2;
	int KernelSize2 = (KernelSize - 1) / 2;
	int min = -max;
	if (cropCloud == false) {
		organizedCloud->width = maxX;
		organizedCloud->height = maxY;
	}
	else {
		organizedCloud->width = maxX - 2 * max;
		organizedCloud->height = maxY - 2 * max;
	}
	organizedCloud->is_dense = false;
	organizedCloud->points.resize(organizedCloud->height * organizedCloud->width);
	pcl::copyPointCloud(*organizedCloud, *organizedCloud2);
	if (cropCloud == false) {
		max = 0;
	}
	size_t size = maxX - max;
	cout << "applyFilterCloud parallel for start" << endl;
	parallel_for(size_t(0), size, [&](size_t ii) {
		//for (int i = max; i < maxX - max; ++i) {
		int i = ii + max;
		for (int j = max; j < maxY - max; ++j) {
			double sum = 0;
			double w = 0.0;
			int ctr = 0;
			int minXGrid = 0;
			int minYGrid = 0;
			int maxXGrid = KernelSize;
			int maxYGrid = KernelSize;
			if (cropCloud == false) {
				sum = 0;
				if (i < KernelSize2) {
					minXGrid = abs(i - KernelSize2);
				}
				if (j < KernelSize2) {
					minYGrid = abs(j - KernelSize2);
				}
				if (abs(i - (maxX - 1)) < KernelSize2) {
					int distToEdge = abs(i - (maxX - 1));
					maxXGrid = (KernelSize - (KernelSize2 - distToEdge));
				}
				if (abs(j - (maxY - 1)) < KernelSize2) {
					int distToEdge = abs(j - (maxY - 1));
					maxYGrid = (KernelSize - (KernelSize2 - distToEdge));
				}
			}
			for (int x = minXGrid; x <= maxXGrid - 1; x++) {
				for (int y = minYGrid; y <= maxYGrid - 1; y++) {
					int ii = i - (KernelSize2 - x);
					int jj = j - (KernelSize2 - y);
					if (cropCloud == false) {
						w += GKernel(x, y) * inputptr->at(ii, jj).z;
						sum = sum + GKernel(x, y);
					}
					else {
						w += GKernel(x, y) * inputptr->at(ii, jj).z;
					}
				}
			}
			if (cropCloud == false && isinf(w / sum)) {
				cout << "applyFilterCloud  FINAL: w: " << w << " sum: " << sum << " ctr: " << ctr << endl;
			}
			if (cropCloud == false) {
				w = w / sum;
			}
			organizedCloud->at(i - max, j - max).x = inputptr->at(i, j).x;
			organizedCloud->at(i - max, j - max).y = inputptr->at(i, j).y;
			if (HighPass) {
				organizedCloud->at(i - max, j - max).z = inputptr->at(i, j).z - w;
				organizedCloud2->at(i - max, j - max).x = inputptr->at(i, j).x;
				organizedCloud2->at(i - max, j - max).y = inputptr->at(i, j).y;
				organizedCloud2->at(i - max, j - max).z = w;
			}
			else {
				organizedCloud->at(i - max, j - max).z = w;
			}
		}
		//	}
		});
	cout << "applyFilterCloud parallel for end" << endl;
	pcl::copyPointCloud(*organizedCloud, *inputptr);
	cout << "applyFilterCloud end" << endl;
}

//Determine the kernel size based one the larger cutoff wavelength that meets accuracy requirement. (better 95%)
int determineKernelSizeFromCutOffWithAccuracyCheck(double cutoffWaveLength) {
	cout << "determineKernelSizeFromCutOffWithAccuracyCheck start" << endl;
	double alpha = 0.4697;
	double pi = 3.14159265358979323846;
	double result = 1;
	double resultNormalized = 1;
	double x = 0;
	int kernelSize = 0;
	double sum = 0;
	while (resultNormalized > 0.05) {
		result = ((1 / (alpha * cutoffWaveLength)) * exp(-pi * pow(((x) / (alpha * cutoffWaveLength)), 2))) *
			((1 / (alpha * cutoffWaveLength)) * exp(-pi * pow(((x) / (alpha * cutoffWaveLength)), 2)));
		sum += result;
		resultNormalized = result / sum;
		x += 0.001;
		kernelSize++;
	}
	if (kernelSize % 2 > 0) {
		kernelSize++;
	}
	cout << "determineKernelSizeFromCutOffWithAccuracyCheck end  kernelSize: " << kernelSize << endl;
	return kernelSize;
}

// determine kernel size based on donwsample value.
int determineKernelSizeFromCutOff(double cutoffWaveLength, float downsampleValue) {
	cout << "determineKernelSizeFromCutOff start" << endl;
	int kernelSize = 0;
	float Lc = 0.5;
	float maxKernelDist = Lc * cutoffWaveLength;
	kernelSize = ceil(maxKernelDist / downsampleValue) * 2 + 1;
	cout << "determineKernelSizeFromCutOff end kernelSize: " << kernelSize << endl;
	return kernelSize;
}

//create a grid which is gauss filtered
Polyhedron gcreateGaussGridMesh(PointList& points, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudXYZ, Polyhedron output_mesh_buffer, int loopCounter,
	string dir, string file, double longWavelengthCutoff, double shortWavelengthCutoff) {
	cout << "gcreateGaussGridMesh start" << endl;

	//double shortWavelengthCutoff = 0.01;
	//double longWavelengthCutoff = 0.25;
	pcl::PointCloud<PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<PointXYZRGBA>);
	for (int i = 0; i < points.size(); i++) {
		pcl::PointXYZRGBA p;
		p.x = points[i].first.x();
		p.y = points[i].first.y();
		p.z = points[i].first.z();
		cloud->push_back(p);
	}
	Eigen::Matrix3f eigen_vector[1];
	std::vector<Eigen::Vector4f> centroid_vec;
	eigen_vector[0] = PCAcalc(cloud);
	centroid_vec.push_back(Transformation(cloud, eigen_vector[0]));
	//rotate cloud so that surface is aligned to a plane
	Transformation(cloudXYZ, eigen_vector[0]);
	passfilter2DGrid(cloud, 0.001);
	saveLegacyDiagnosticCloud(cloud, dir, file, "03_gaussian_grid.pcd");
	int KernelSize = determineKernelSizeFromCutOffWithAccuracyCheck(longWavelengthCutoff);
	MatrixXf GKernelLow(KernelSize, KernelSize);
	MatrixXf GKernelHigh(KernelSize, KernelSize);
	GKernelLow = crateGaussFilterKernel(GKernelLow, shortWavelengthCutoff, KernelSize);
	GKernelHigh = crateGaussFilterKernel(GKernelHigh, longWavelengthCutoff, KernelSize);
	//apply filter that changes the z values. 
	applyFilterKernelToPc(cloud, GKernelLow, GKernelHigh, cloudXYZ, KernelSize);
	saveLegacyDiagnosticCloud(cloud, dir, file, "04_gaussian_filtered.pcd");
	TransformBack(cloud, eigen_vector[0], centroid_vec[0]);
	TransformBack(cloudXYZ, eigen_vector[0], centroid_vec[0]);
	PointList points2;
	for (int i = 0; i < cloud->size(); i++) {
		Point pt(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
		Point_with_normal pn;
		pn.first = pt;
		points2.push_back(pn);
	}
	//create a mesh from filtered grid coud.
	output_mesh_buffer = createDenseMesh(output_mesh_buffer, points2, loopCounter, dir, file);
	if (legacyDiagnosticsEnabled()) {
		std::ofstream meshFile(legacyDiagnosticPath(dir, file, "05_gaussian_mesh.off"));
		meshFile << output_mesh_buffer;
		std::cout << "[legacy] gaussian_mesh_vertices: " << output_mesh_buffer.size_of_vertices()
			<< ", faces: " << output_mesh_buffer.size_of_facets() << std::endl;
	}
	cout << "gcreateGaussGridMesh done" << endl;
	return output_mesh_buffer;
}

// apply gaussian filter to point cloud which has not been aligned to a xy grid. Slower, currently not 100% sure its correct. 
std::vector<double> calcDistFromLocalGauss(PointList& points, PointCloud<PointXYZRGBA>::Ptr& cloud2, std::vector<double>& dist, size_t n, std::vector<double>& varVec,
	std::vector<double>& sumVec, std::vector<double>& CtrVec, int PointsOnVariogram, double span, string dir, string file, int j,
	double shortWavelengthCutoff, double longWavelengthCutoff) {
	cout << "calcDistFromLocalGauss start" << endl;
	//std::stringstream ss2;
	//ss2 << dir << file << "gaussDistBeginning" << j << ".pcd";
	//pcl::io::savePCDFileASCII(ss2.str(), *cloud2);
	pcl::PointCloud<PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<PointXYZRGBA>);
	pcl::PointCloud<PointXYZRGBA>::Ptr cloud3(new pcl::PointCloud<PointXYZRGBA>);
	pcl::copyPointCloud(*cloud2, *cloud);
	pcl::copyPointCloud(*cloud, *cloud3);
	Eigen::Matrix3f eigen_vector[1];
	std::vector<Eigen::Vector4f> centroid_vec;
	eigen_vector[0] = PCAcalc(cloud);
	centroid_vec.push_back(Transformation(cloud, eigen_vector[0]));
	const int NmbrOfPtInPC = points.size();
	size_t size = points.size();
	std::vector<std::vector<double> > sumVecParallel(size, std::vector<double>(PointsOnVariogram + 1));
	std::vector<std::vector<double> > CtrVecParallel(size, std::vector<double>(PointsOnVariogram + 1));
	std::vector<double> SvrPoint(cloud->size(), 0);
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//Create Kd tree for fast distance calculations                                               
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(cloud);
	double sum = 0.0;
	double alpha = 0.4697;
	//double cutoffWaveLength = 0.0025;
	double pi = 3.14159265358979323846;
	double yVal = ((1 / (alpha * shortWavelengthCutoff)) * exp(-pi * pow((0 / (alpha * shortWavelengthCutoff)), 2)));
	//go through all points and determine distances to underlying geometry based on surrounding points and wheights.
	parallel_for(size_t(0), size, [&](size_t i) {
		//for (int i = 0; i < size; i++) {
		pcl::PointXYZRGBA searchPoint = cloud->points[i];
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;
		float radius = PointsOnVariogram * span;
		kdtree.radiusSearch(searchPoint, shortWavelengthCutoff, pointIdxRadiusSearch, pointRadiusSquaredDistance);
		double sum = 0;
		double w = 0;
		for (int j = 1; j < pointIdxRadiusSearch.size(); j++) {
			double s = ((1 / (alpha * shortWavelengthCutoff)) * exp(-pi * pow((sqrt(pointRadiusSquaredDistance[j]) / (alpha * shortWavelengthCutoff)), 2)));// *yVal;
			w += s * cloud->points[pointIdxRadiusSearch[j]].z;
			sum += s;
		}
		dist[i] = w / sum;
		cloud3->points[i].z = dist[i];
		});

	//(1 / (pow(alpha, 2) * pow(cutoffWaveLength, 2))) * exp((-pi / pow(alpha, 2)) * ((pow((x * factor), 2) + pow((y * factor), 2)) / pow(cutoffWaveLength, 2)));
	kdtree.setInputCloud(cloud3);
	yVal = ((1 / (alpha * longWavelengthCutoff)) * exp(-pi * pow((0 / (alpha * longWavelengthCutoff)), 2)));
	parallel_for(size_t(0), size, [&](size_t i) {
		//for (int i = 0; i < size; i++) {
		pcl::PointXYZRGBA searchPoint = cloud3->points[i];
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;
		float radius = PointsOnVariogram * span;
		kdtree.radiusSearch(searchPoint, longWavelengthCutoff, pointIdxRadiusSearch, pointRadiusSquaredDistance);
		double sum = 0;
		double w = 0;
		for (int j = 1; j < pointIdxRadiusSearch.size(); j++) {
			double s = ((1 / (alpha * longWavelengthCutoff)) * exp(-pi * pow((sqrt(pointRadiusSquaredDistance[j]) / (alpha * longWavelengthCutoff)), 2)));// *yVal;
			w += s * dist[pointIdxRadiusSearch[j]];
			sum += s;
		}
		dist[i] = w / sum;
		});
	//}

	TransformBack(cloud, eigen_vector[0], centroid_vec[0]);
	pcl::copyPointCloud(*cloud, *cloud2);
	PointList points2;
	for (int i = 0; i < cloud3->size(); i++) {
		cloud3->points[i].z = dist[i];
	}
	colorPcBasedOnDist(dist, cloud3);
	cout << "calcDistFromLocalGauss start" << endl;
	return SvrPoint;
}

//first creates a dense mesh then remeshes it into a sparser one.
int globalMeshingVariogram(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer, PointList& points, PointList& points2,
	bool bezier, int j, std::string& dir, std::string& file, double target_edge_length, std::vector<double>& dist,
	std::chrono::steady_clock::time_point& begin, std::vector<double>& SvrPoint, /*double* Svr,*/ std::vector<double>& varVec, std::vector<double>& sumVec, std::vector<double>& CtrVec,
	int PointsOnVariogram, double span, bool& retflag, std::vector<double>& SaSqSvr, bool GaussDistance, bool gaussGrd,
	int RoughnessParameterSaSqSvr, bool OverWriteParameters, double longWavelengthCutoff, double shortWavelengthCutoff)
{
	cout << "globalMeshingVariogram start" << endl;
	std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudOnSurface(new pcl::PointCloud<pcl::PointXYZRGBA>);
	size_t n = points2.size();
	std::vector<double> dist2(n, 1);
	if (GaussDistance) {
		for (int i = 0; i < points2.size(); i++) {
			pcl::PointXYZRGBA p;
			p.x = points2[i].first.x();
			p.y = points2[i].first.y();
			p.z = points2[i].first.z();
			cloudOnSurface->push_back(p);
		}
		//determine distance value
		SvrPoint = calcDistFromLocalGauss(points2, cloudOnSurface, dist2, n,/* Svr,*/ varVec, sumVec, CtrVec, PointsOnVariogram, span, dir, file, j,
			shortWavelengthCutoff, longWavelengthCutoff);
		dist = dist2;
	}
	else {
		retflag = true;
		Polyhedron mesh;
		//create mesh from bezier
		if (bezier) {
			createMeshBezier(points, j, dir, file, target_edge_length);
		}
		//create mesh from smoothed gauss grid
		else if (gaussGrd) {
			PointCloud<PointXYZRGBA>::Ptr cloud_cut(new PointCloud<PointXYZRGBA>());
			for (int i = 0; i < points2.size(); i++) {
				pcl::PointXYZRGBA p;
				p.x = points2[i].first.x();
				p.y = points2[i].first.y();
				p.z = points2[i].first.z();
				cloud_cut->push_back(p);
			}
			mesh = gcreateGaussGridMesh(points, cloud_cut, mesh, j, dir, file, longWavelengthCutoff, shortWavelengthCutoff);
			points2.clear();
			for (int i = 0; i < cloud_cut->size(); i++) {
				Point pt(cloud_cut->points[i].x, cloud_cut->points[i].y, cloud_cut->points[i].z);
				Point_with_normal pn;
				pn.first = pt;
				points2.push_back(pn);
			}
			n = points2.size();

		}
		//create mesh from dense mesh and then remeshing to target size.
		else {
			mesh = createDenseMesh(mesh, points, j, dir, file);
			mesh = remesh(mesh, j, target_edge_length, dir, file);

		}
		// constructs the AABB tree and the internal search tree for efficient distance queries.
		Tree tree(CGAL::faces(mesh).first, CGAL::faces(mesh).second, mesh);
		tree.accelerate_distance_queries();
		cloud_buffer->clear();
		for (int i = 0; i < points2.size(); i++) {
			pcl::PointXYZRGBA p;
			p.x = points2[i].first.x();
			p.y = points2[i].first.y();
			p.z = points2[i].first.z();
			cloudOnSurface->push_back(p);
			cloud_buffer->push_back(p);
		}
		//calculate distance point cloud to mesh
		/*dist2 =*/ ParallelDistance(dist2, points2, cloudOnSurface, &tree, n);
		pcl::PointCloud<PointXYZRGBA>::Ptr cloud3(new pcl::PointCloud<PointXYZRGBA>);
		pcl::copyPointCloud(*cloudOnSurface, *cloud3);
		for (int i = 0; i < cloud3->size(); i++) {
			cloud3->points[i].z = dist2[i];
		}
		std::stringstream ss1;
		ss1 << dir << file << "globalMeshingVariogramDistance" << j << ".pcd";
		pcl::io::savePCDFileASCII(ss1.str(), *cloud3);
		//color point cloud
		colorPcBasedOnDist(dist2, cloud_buffer, dir + file, 1);
		cloud_buffer->clear();
		PointList bufferPoints2 = points2;
		points2.clear();
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudOnSurfaceBuffer(new pcl::PointCloud<pcl::PointXYZRGBA>);
		//Change format clouds
		pcl::copyPointCloud(*cloudOnSurface, *cloudOnSurfaceBuffer);
		cloudOnSurface->clear();
		for (int i = 0; i < cloudOnSurfaceBuffer->size(); i++) {
			if (dist2[i] != 1 && dist2[i] != -1) {
				Point pt(cloudOnSurfaceBuffer->points[i].x, cloudOnSurfaceBuffer->points[i].y, cloudOnSurfaceBuffer->points[i].z);
				Point_with_normal pn;
				pn.first = pt;
				cloudOnSurface->push_back(cloudOnSurfaceBuffer->points[i]);
				dist.push_back(dist2[i]);
				points2.push_back(bufferPoints2[i]);
				pcl::PointXYZRGBA p;
				p.x = bufferPoints2[i].first.x();
				p.y = bufferPoints2[i].first.y();
				p.z = bufferPoints2[i].first.z();
				cloud_buffer->push_back(p);
			}
		}
		colorPcBasedOnDist(dist, cloud_buffer, dir + file, 2);
		if (saveClouds) {
			pcl::io::savePCDFileASCII(dir + file + "Variogram.pcd", *cloudOnSurface);
		}

	}
	//Calculate roughness
	double sa = 0;
	double sq = 0;
	double svr = 0;
	//Sa
	if (RoughnessParameterSaSqSvr == 0 || OverWriteParameters == true) {
		double sumSa = 0;
		int ctr = 0;
		for (int j = 0; j < dist.size(); j++) {
			sumSa += abs(dist[j]);
			ctr++;
		}
		CtrVec[0] = ctr;
		sumVec[0] = sumSa;
		sa = (sumSa / ctr) * 1000000;
	}
	//Sq
	if (RoughnessParameterSaSqSvr == 1 || OverWriteParameters == true) {
		double sumSq = 0;
		int ctr = 0;
		for (int j = 0; j < dist.size(); j++) {
			if (j < 10) {
				cout << "dist[j]: " << dist[j] << "   pow(dist[j], 2): " << pow(dist[j], 2) << endl;
			}
			sumSq += pow(dist[j], 2);
			ctr++;
		}
		CtrVec[0] = ctr;
		sumVec[0] = sumSq;
		sq = sqrt(sumSq / ctr) * 1000000;
		cout << "CtrVec[0]: " << CtrVec[0] << endl;
		cout << "sumVec[0]: " << sumVec[0] << endl;
		cout << "sq: " << sq << endl;
	}
	//Svr
	if (RoughnessParameterSaSqSvr == 2 || OverWriteParameters == true) {
		//Calculate variogram roughness
		/*std::vector<double>*/svr = calcVariogramFromPCParallel(SvrPoint,/*mesh, points2, pointsOnSurface, */cloudOnSurface, dist,/* Svr,*/ /*edgePoints,*/ varVec, sumVec, CtrVec,
			PointsOnVariogram, span);
	}
	cout << "SaSqSvr.size(): " << SaSqSvr.size() << endl;
	cout << "sq: " << sq << endl;
	SaSqSvr.push_back(sa);
	SaSqSvr.push_back(sq);
	SaSqSvr.push_back(svr * 1000);
	cout << "Sa: " << SaSqSvr[0] << endl;
	cout << "Sq: " << SaSqSvr[1] << endl;
	cout << "Svr: " << SaSqSvr[2] << endl;
	retflag = false;
	std::cout << "Time for code execution globalMeshingVariogram= " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin2).count() << "[ms]" << std::endl;

	return {};
}

//combine results from multiple runs
void clusterFinish(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBACombined,
	std::vector<std::vector<double>>& sumMatrix, std::vector<double>& sumVec, std::vector<std::vector<double>>& CtrMatrix, std::vector<double>& CtrVec,
	std::vector<std::vector<double>>& varMatrix, std::vector<double>& varVec, std::vector<double>& SvrPoint, int& j,
	std::vector<double>& SvrPointCombined, std::vector<double>& dist2Combined, std::vector<double>& dist, int RoughnessParameterSaSqSvr)
{
	cout << "clusterFinish start" << endl;
	sumMatrix.push_back(sumVec);
	CtrMatrix.push_back(CtrVec);
	varMatrix.push_back(varVec);
	if (RoughnessParameterSaSqSvr == 2) {
		colorPcBasedOnLocalVariogram(SvrPoint, cloud_buffer);
	}
	j++;
	*cloudRGBACombined += *cloud_buffer;
	SvrPointCombined.insert(SvrPointCombined.end(), SvrPoint.begin(), SvrPoint.end());
	dist2Combined.insert(dist2Combined.end(), dist.begin(), dist.end());
	cout << "clusterFinish end" << endl;
}

//create an output result table
void createResultTable(std::string& dir, std::string& file, std::vector<std::vector<double>>& varMatrix, double downsample,
	double cell_size_grid_simplify_point_set, int PointsOnVariogram, double span, double target_edge_length, std::vector<double>& varVec,
	double& SvrFinal, double time, int PCsize, std::vector<double> SaSqSvr, int RoughnessParameterSaSqSvr)
{
	cout << "createTable start" << endl;
	//Save csv file
	string buff = dir + file + "Results.csv";
	cout << "Operator: " << Operator << " Casting: " << Casting << " Run: " << Run << " GageRR: " << GageRR << endl;
	if (GageRR) {
		cout << "within if" << endl;
		std::stringstream buffss;
		buffss << "C:\\Users\\dschimpf\\Box\\Daniel Schimpf\\RE\\Surface Project\\PCL_Project\\GageRR\\Gage_Results_O" << Operator << "_C" << Casting << "_R" << Run << ".csv";
		std::string s = buffss.str();
		//string buffstr = "C:\\Users\\dschimpf\\Box\\Daniel Schimpf\\RE\\Surface Project\\PCL_Project\\GageRR\\Gage" + "_O" + Operator + "_C" + Casting + "_R" + Run +".pcd";
		buff = buffss.str();
	}
	std::ofstream myfile;
	myfile.open(buff);
	myfile << file << "\n";
	myfile << "Downsample:," << downsample << "\n";
	myfile << "Downsample Mesh:," << cell_size_grid_simplify_point_set << "\n";
	myfile << "PointsOnVariogram:," << PointsOnVariogram << "\n";
	myfile << "Span:," << span << "\n";
	myfile << "Target_edge_length:," << target_edge_length << "\n";
	myfile << "Execution Time:," << time << "\n";
	myfile << "Point Cloud Size:," << PCsize << "\n";
	myfile << "Sa:," << SaSqSvr[0] << "\n";
	myfile << "Sq:," << SaSqSvr[1] << "\n";
	myfile << "Svr:," << SaSqSvr[2] << "\n";
	myfile << "Reported Roughness Value:," << SaSqSvr[RoughnessParameterSaSqSvr] << "\n";
	myfile << "Overall Roughness \n";
	for (int i = 0; i < PointsOnVariogram; i++) {
		myfile << "Svr[" << i << "]," << varVec[i] << "\n";
	}
	myfile << "Svr [mm]:," << SaSqSvr[2] << endl;
	double sumSvr = 0;
	cout << "Svr[mm]: " << SaSqSvr[2] << endl;
	myfile << "\n";
	myfile << "Individual Roughness of sections #\n";
	for (int j = 0; j < varMatrix.size(); j++) {
		myfile << "," << j + 1;
	}
	myfile << "\n";
	for (int i = 0; i < PointsOnVariogram; i++) {
		//cout << i;
		myfile << "Svr[" << i << "]";
		for (int j = 0; j < varMatrix.size(); j++) {
			//cout << ".";
			myfile << "," << varMatrix[j][i];
		}
		myfile << "\n";
		cout << " " << endl;
	}
	myfile << "Svr [mm]:";
	for (int j = 0; j < varMatrix.size(); j++) {
		sumSvr = 0;
		for (int i = 0; i < PointsOnVariogram; i++) {
			sumSvr += varMatrix[j][i];
		}
		myfile << "," << sumSvr / PointsOnVariogram;
	}
	myfile.close();
	cout << "createTable end" << endl;
}

//seperate a large point cloud into smaller clusters
void clusterPc(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& edgesRemoved, std::vector<pcl::PointIndices>& cluster_indices)
{
	cout << "clusterCloud start" << endl;
	//SEGMENT POINT CLOUD
	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr treeSegment(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	treeSegment->setInputCloud(edgesRemoved);
	pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
	ec.setClusterTolerance(0.001); // 2cm
	ec.setMinClusterSize(5000);
	ec.setMaxClusterSize(10000000);
	ec.setSearchMethod(treeSegment);
	ec.setInputCloud(edgesRemoved);
	ec.extract(cluster_indices);
	cout << "clusterCloud end" << endl;
}

//the fun with the variogram begins. this sets everything up so that the underlying geometry can be determined and used to calculate the variogram roughness.
int prepareVariogramCalculation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_buffer, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudRGBACombined,
	std::string& dir, std::string& file, bool statFilter, bool GridFilterBool, bool removeEdgesAfterMeshing, double downsample,
	double cell_size_grid_simplify_point_set, pcl::PointXYZRGBA& outCentroid2, bool globalMeshing, bool bezier, double target_edge_length,
	std::chrono::steady_clock::time_point& begin, /*double* Svr,*/ std::vector<double>& varVec, int PointsOnVariogram, double span, std::vector<std::vector<double>>& varMatrix,
	std::vector<double>& SvrPointCombined, std::vector<double>& dist2Combined, bool& retflag, bool GaussDistance, bool gaussGrd, std::vector<double>& SaSqSvr,
	int RoughnessParameterSaSqSvr, bool OverWriteParameters, double longWavelengthCutoff, double shortWavelengthCutoff)
{
	cout << "prepareVariogramCalculation start" << endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBA(new pcl::PointCloud<pcl::PointXYZRGBA>);
	retflag = true;
	std::vector<double> sumVec(PointsOnVariogram, 0);
	std::vector<double> CtrVec(PointsOnVariogram, 0);
	std::vector<std::vector<double>> sumMatrix;
	std::vector<std::vector<double>> CtrMatrix;
	int k = 0;
	PointList points;
	PointList points2;
	std::vector<double> SvrPoint;
	std::vector<double> dist;
	Prep(cloud_buffer, outCentroid2, points, points2, k, dir, file, statFilter, GridFilterBool, removeEdgesAfterMeshing, downsample, cell_size_grid_simplify_point_set);
	if (legacyDiagnosticsEnabled()) {
		std::cout << "[legacy] prep_distance_points: " << points2.size() << std::endl;
		std::cout << "[legacy] prep_mesh_points: " << points.size() << std::endl;
	}
	if (globalMeshing) {
		// most important line in this funcion.
		int retval = globalMeshingVariogram(cloud_buffer, points, points2, bezier, k, dir, file, target_edge_length, dist, begin, SvrPoint,
			varVec, sumVec, CtrVec, PointsOnVariogram, span, retflag, SaSqSvr, GaussDistance, gaussGrd, RoughnessParameterSaSqSvr, OverWriteParameters,
			longWavelengthCutoff, shortWavelengthCutoff);
		if (retflag) return -1;
	}
	else {
		SvrPoint = calcVariogramFromPCParallel_local(points2, varVec, sumVec, CtrVec, PointsOnVariogram, span, false);
	}
	clusterFinish(cloud_buffer, cloudRGBACombined, sumMatrix, sumVec, CtrMatrix, CtrVec, varMatrix, varVec, SvrPoint, k,
		SvrPointCombined, dist2Combined, dist, RoughnessParameterSaSqSvr);
	//Combine Svr Values
	for (int i = 0; i < PointsOnVariogram; i++) {
		double Ctr = 0;
		double sum = 0;
		for (int j = 0; j < sumMatrix.size(); j++) {
			sum = sum + sumMatrix[j][i];
			Ctr = Ctr + CtrMatrix[j][i];
		}
		varVec[i] = sqrt((1 / (2 * Ctr)) * sum);
	}
	retflag = false;
	cout << "prepareVariogramCalculation end" << endl;
	return {};
}

int RoughnessCalcCluster(std::vector<pcl::PointIndices>& cluster_indices, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& edgesRemoved, std::string& dir, std::string& file,
	bool statFilter, bool GridFilterBool, bool removeEdgesAfterMeshing, double downsample, double cell_size_grid_simplify_point_set,
	pcl::PointXYZRGBA& outCentroid2, bool globalMeshing, bool bezier, double target_edge_length, std::chrono::steady_clock::time_point& begin, /*double* Svr,*/
	std::vector<double>& varVec, int PointsOnVariogram, double span, std::vector<std::vector<double>>& varMatrix, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloudRGBACombined,
	std::vector<double>& SvrPointCombined, std::vector<double>& dist2Combined, bool& retflag, bool GaussDistance, bool gaussGrd, std::vector<double>& SaSqSvr, int RoughnessParameterSaSqSvr,
	bool OverWriteParameters, double longWavelengthCutoff, double shortWavelengthCutoff)
{
	cout << "RoughnessCalcCluster start " << endl;
	PointCloud<PointXYZRGBA>::Ptr cloud_buffer(new PointCloud<PointXYZRGBA>());
	std::chrono::steady_clock::time_point beginVar = std::chrono::steady_clock::now();
	std::vector<double> sumVec(PointsOnVariogram, 0);
	std::vector<double> CtrVec(PointsOnVariogram, 0);
	std::vector<std::vector<double>> sumMatrix;
	std::vector<std::vector<double>> CtrMatrix;
	retflag = true;
	//outputClusterCommandLine(cluster_indices, edgesRemoved);
	int k = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		PointList points;
		PointList points2;
		std::vector<double> SvrPoint;
		std::vector<double> dist;
		clusterPrep(k, it, edgesRemoved, dir, file, statFilter, GridFilterBool, removeEdgesAfterMeshing, downsample, cloud_buffer, points2, cell_size_grid_simplify_point_set, points, outCentroid2);
		if (globalMeshing) {
			int retval = globalMeshingVariogram(cloud_buffer, points, points2, bezier, k, dir, file, target_edge_length, dist, begin, SvrPoint,
				varVec, sumVec, CtrVec, PointsOnVariogram, span, retflag, SaSqSvr, GaussDistance, gaussGrd, RoughnessParameterSaSqSvr, OverWriteParameters,
				longWavelengthCutoff, shortWavelengthCutoff);
			if (retflag) return retval;
		}
		else {
			SvrPoint = calcVariogramFromPCParallel_local(points2, varVec, sumVec, CtrVec, PointsOnVariogram, span, false);
		}
		clusterFinish(cloud_buffer,/*cloudRGBA,*/cloudRGBACombined, sumMatrix, sumVec, CtrMatrix, CtrVec, varMatrix, varVec, SvrPoint, k, SvrPointCombined, dist2Combined, dist, RoughnessParameterSaSqSvr);
	}
	//if (saveClouds) {
	//	pcl::io::savePCDFileASCII(dir + file + "_AR_ColoredCombined.pcd", *cloudRGBACombined);
	//}
	if (RoughnessParameterSaSqSvr == 0 || RoughnessParameterSaSqSvr == 1) {
		double ctr = 0;
		double sum = 0;
		for (int j = 0; j < sumMatrix.size(); j++) {
			sum = sum + sumMatrix[j][0];
			ctr = ctr + CtrMatrix[j][0];
		}
		if (RoughnessParameterSaSqSvr == 0) {
			SaSqSvr[0] = (sum / ctr) * 1000000;
		}
		else {
			SaSqSvr[1] = sqrt(sum / ctr) * 1000000;
		}
	}
	else if (RoughnessParameterSaSqSvr == 2) {
		//Combine Svr Values
		for (int i = 0; i < PointsOnVariogram; i++) {
			double Ctr = 0;
			double sum = 0;
			for (int j = 0; j < sumMatrix.size(); j++) {
				sum = sum + sumMatrix[j][i];
				Ctr = Ctr + CtrMatrix[j][i];
			}
			varVec[i] = sqrt((1 / (2 * Ctr)) * sum);
		}
	}
	retflag = false;
	cout << "RoughnessCalcCluster end " << endl;
	return {};
}

bool LoadInputFromFile(std::string& fileEnding, std::string& filename, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, boost::shared_ptr< ::pcl::PolygonMesh>  inputMesh, bool stlInput)
{
	cout << "LoadInputFromFile start " << endl;
	if (fileEnding.compare(".pcd") == 0) {
		if (pcl::io::loadPCDFile(filename, *cloud) < 0) {
			std::cout << "Error loading point cloud " << filename << std::endl << std::endl;
			//return -1;
		}
		std::cout << "Loaded point cloud " << filename << std::endl << std::endl;
	}
	else if (fileEnding.compare(".ply") == 0) {
		if (pcl::io::loadPLYFile(filename, *cloud) < 0) {
			std::cout << "Error loading point cloud " << filename << std::endl << std::endl;
			//return -1;
		}
		std::cout << "Loaded point cloud " << filename << std::endl << std::endl;
	}
	else if (fileEnding.compare(".asc") == 0 || fileEnding.compare(".txt") == 0) {
		loadAsciCloud(filename, cloud);
		std::cout << "Loaded point cloud " << filename << std::endl << std::endl;
	}
	else if (fileEnding.compare(".stl") == 0) {
		pcl::io::loadPolygonFileSTL(filename, *inputMesh);
		pcl::PCLPointCloud2 pc2;
		pc2 = inputMesh->cloud;
		pcl::fromPCLPointCloud2(pc2, *cloud);
		std::cout << "Loaded stl " << filename << std::endl << std::endl;
		stlInput = true;
		cout << "loading   stlInput: " << stlInput << endl;
	}
	cout << "LoadInputFromFile end " << endl;
	return stlInput;
}

void CenterPointCloudAndAdjustUnits(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, pcl::PointXYZRGBA& outCentroid2, bool UnitMM, bool UnitInch)
{
	cout << "CenterPointCloudAndAdjustUnits start " << endl;
	pcl::computeCentroid(*cloud, outCentroid2);
	Eigen::Affine3f transform2 = Eigen::Affine3f::Identity();
	transform2.translation() << -outCentroid2.x, -outCentroid2.y, -outCentroid2.z;
	pcl::transformPointCloud(*cloud, *cloud, transform2);
	//Some of the parameters set are assuming a certain length unit. This converts mm to m. 
	if (UnitMM == true) {
		for (size_t i = 0; i < cloud->size(); ++i)
		{
			cloud->points[i].x *= 0.001;
			cloud->points[i].y *= 0.001;
			cloud->points[i].z *= 0.001;
		}
	}
	else if (UnitInch == true) {
		for (size_t i = 0; i < cloud->size(); ++i)
		{
			cloud->points[i].x *= 0.0254;
			cloud->points[i].y *= 0.0254;
			cloud->points[i].z *= 0.0254;
		}
	}
	pcl::computeCentroid(*cloud, outCentroid2);
	cout << "CenterPointCloudAndAdjustUnits end " << endl;
}

void DetermineEdgesAndCluster(std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud,
	std::string& dir,
	std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& edgesRemoved,
	bool removeEdgesAfterMeshing,
	std::vector<pcl::PointIndices>& cluster_indices) {
	std::cout << "DetermineEdgesAndCluster start" << std::endl;

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr edgesCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

	// Copy cloud for edge detection
	pcl::copyPointCloud(*cloud, *edgesCloud);
	EdgeDetection(edgesCloud, dir);

	// Collect edge points (based on color criteria)
	for (size_t i = 0; i < edgesCloud->size(); i++) {
		if (edgesCloud->points[i].r >= 239) {
			inliers->indices.push_back(i);
		}
	}

	// Copy detected edges to edgesRemoved
	pcl::copyPointCloud(*edgesCloud, *edgesRemoved);

	if (!removeEdgesAfterMeshing) {
		auto stdEdgesRemoved = std::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGBA>>(
			edgesRemoved.get(), [](pcl::PointCloud<pcl::PointXYZRGBA>*) {});

		pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
		extract.setInputCloud(stdEdgesRemoved);
		extract.setIndices(inliers);
		extract.setNegative(false);
		extract.filter(*edgesRemoved);
	}

	if (!edgesRemoved->empty()) {
		pcl::copyPointCloud(*edgesRemoved, *cloud);
	}

	// Convert boost::shared_ptr to std::shared_ptr for clusterPc
	auto stdEdgesRemoved = std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>(
		edgesRemoved.get(), [](pcl::PointCloud<pcl::PointXYZRGBA>*) {});

	// Perform clustering
	clusterPc(stdEdgesRemoved, cluster_indices);

	std::cout << "DetermineEdgesAndCluster end. Number of clusters: " << cluster_indices.size() << std::endl;
}

void VoxelGridDownsample(std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>>& cloud, double downsample) {
	std::cout << "VoxelGridDownsample start cloud->size(): " << cloud->size() << std::endl;

	pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(downsample, downsample, downsample);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	vg.filter(*filteredCloud);

	pcl::copyPointCloud(*filteredCloud, *cloud);
	std::cout << "VoxelGridDownsample end cloud->size(): " << cloud->size() << std::endl;
}

//Performs point cloud tranfromation base on the eig_vec
void transformPc(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, Eigen::Matrix3f eig_vec)
{
	cout << "transformPc start " << endl;
	Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
	// Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			transform_1(j, i) = eig_vec(i, j);
		}
	}
	pcl::transformPointCloud(*inputptr, *inputptr, transform_1);
	cout << "transformPc end " << endl;
}

double distance(pcl::PointXYZRGBA Old, pcl::PointXYZRGBA New) {
	return sqrt(pow(Old.x - New.x, 2) + pow(Old.y - New.y, 2) + pow(Old.z - New.z, 2));
}

double distance(Point Old, Point New) {
	return sqrt(pow(Old.x() - New.x(), 2) + pow(Old.y() - New.y(), 2) + pow(Old.z() - New.z(), 2));
}

//The meshing operation sometimes creates way to large triangles to connect far off points. These are removed for a more accurate representation.
Polyhedron deleteLargeTriangles(Polyhedron meshInput) {
	cout << "deleteLargeTriangles start " << endl;
	for (Polyhedron::Facet_iterator face = meshInput.facets_begin(); face != meshInput.facets_end(); ++face) {
		Polyhedron::Halfedge_const_handle begin = face->halfedge();
		Polyhedron::Halfedge_const_handle edge = begin;
		double d = 0;
		Point ptNew;
		Point ptOld;
		int i = 0;
		do {
			ptNew = edge->vertex()->point();
			if (i > 0) {
				d = d + distance(ptOld, ptNew);
			}
			ptOld = ptNew;
			edge = edge->next();
			i++;
		} while (edge != begin);
		if (d > 0.002) {
			meshInput.erase_facet(face->halfedge());
		}
	}
	cout << "deleteLargeTriangles end " << endl;
	return meshInput;
}

//The form of a plan is removed from the point cloud, by aligning it with the best matching plane
void formRemovalPlane(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr) {
	cout << "formRemoval start size: " << inputptr->size() << endl;
	Eigen::Matrix3f eigen_vector;
	std::vector<Eigen::Vector4f> centroid_vec;
	eigen_vector = PCAcalc(inputptr);
	transformPc(inputptr, eigen_vector);
	cout << "formRemoval end " << endl;
}

// Part of the meshing operation. First create a very dense mesh
Polyhedron denseMesh(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr) {
	cout << "denseMesh start size: " << inputptr->size() << endl;
	Polyhedron output_mesh_buffer;
	PointList points;
	Point_set PointsSet;
	PointsSet.add_normal_map();
	for (int i = 0; i < inputptr->size(); i++) {
		Point pt(inputptr->points[i].x, inputptr->points[i].y, inputptr->points[i].z);
		Point_with_normal pn;
		pn.first = pt;
		Point_set::iterator new_item = PointsSet.insert(pn.first, pn.second);
		points.push_back(pn);
	}
	typedef std::array<std::size_t, 3> Facet; // Triple of indices
	std::vector<Facet> facets;
	double beta = 0.52;
	double radius_ratio_bound = 5 * 4;
	//create polygons from point cloud
	cout << "denseMesh  advancing_front_surface_reconstruction start" << endl;
	CGAL::advancing_front_surface_reconstruction(PointsSet.points().begin(), PointsSet.points().end(), std::back_inserter(facets), radius_ratio_bound, beta);
	cout << "denseMesh  advancing_front_surface_reconstruction end" << endl;
	// copy points for random access
	std::vector<Point> vertices;
	vertices.reserve(PointsSet.points().size());
	std::copy(PointsSet.points().begin(), PointsSet.points().end(), std::back_inserter(vertices));
	cout << "denseMesh  polygon_soup_to_polygon_mesh start" << endl;
	//create mesh from polygon
	CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(vertices, facets, output_mesh_buffer);
	cout << "denseMesh  polygon_soup_to_polygon_mesh end" << endl;

	//remove bad (large) polygons
	output_mesh_buffer = deleteLargeTriangles(output_mesh_buffer);
	if (saveClouds) {
		std::stringstream ss;;
		ss << dir << file << "denseGrid.off";
		std::ofstream f(ss.str().c_str());
		f << output_mesh_buffer;
		f.close();
	}
	cout << "denseMesh end" << endl;
	return output_mesh_buffer;
}

//Sample points from a mesh on a grid. This grid of points is optimal for gaussian filter application. 
void samplePointsFromMesh(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, Polyhedron polyhedron, double downsampleValue, std::vector<double>& gridParameters) {
	cout << "samplePointsFromMesh start size: " << inputptr->size() << endl;
	// constructs AABB tree
	Tree tree(faces(polyhedron).first, faces(polyhedron).second, polyhedron);
	// constructs segment query
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr organizedCloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointXYZRGBA minPt, maxPt;
	pcl::PointXYZRGBA p;
	//determine grid parameters
	pcl::getMinMax3D(*inputptr, minPt, maxPt);
	int stepsX = ceil((maxPt.x - minPt.x) / downsampleValue) + 1;
	int stepsY = ceil((maxPt.y - minPt.y) / downsampleValue) + 1;
	gridParameters.push_back(maxPt.x);
	gridParameters.push_back(maxPt.y);
	gridParameters.push_back(stepsX);
	gridParameters.push_back(stepsY);
	Eigen::Vector4f centroid;
	organizedCloud->width = stepsX;
	organizedCloud->height = stepsY;
	organizedCloud->is_dense = false;
	organizedCloud->points.resize(organizedCloud->height * organizedCloud->width);
	size_t size = stepsX;
	cout << "samplePointsFromMesh parallel for start" << endl;
	parallel_for(size_t(0), size, [&](size_t ii) {
		//for (int ix = 0; ix < stepsX; ix++) {
		int ix = ii;
		for (int iy = 0; iy < stepsY; iy++) {
			double x = minPt.x + ix * downsampleValue;
			double y = minPt.y + iy * downsampleValue;
			Point a(x, y, -1);
			Point b(x, y, 1);
			Segment segment_query(a, b);
			std::list<Segment_intersection> intersections;
			tree.all_intersections(segment_query, std::back_inserter(intersections));
			auto intersection2 = intersections.begin();
			double sum = 0;
			int ctr = 0;
			for (int i = 0; i < tree.number_of_intersected_primitives(segment_query); i++) {
				Segment_intersection intersection = *intersection2;
#if defined(CGAL_VERSION_NR) && CGAL_VERSION_NR >= 1060000000
				const Point* p = std::get_if<Point>(&intersection->first);
#else
				const Point* p = boost::get<Point>(&intersection->first);
#endif
				if (p) {
					sum = p->z();
					ctr++;
				}
				if (i < tree.number_of_intersected_primitives(segment_query)) {
					std::advance(intersection2, 1);
				}
			}
			organizedCloud->at(ix, iy).x = x;
			organizedCloud->at(ix, iy).y = y;
			if (ctr != 0) {
				if (ctr > 1) {
					//no z value found
					organizedCloud->at(ix, iy).z = 1;
				}
				else {
					//new z value
					organizedCloud->at(ix, iy).z = sum / ctr;
				}
			}
			else {
				//no z value found
				organizedCloud->at(ix, iy).z = 1;
			}
		}
		//}
		});
	cout << "samplePointsFromMesh parallel for end" << endl;
	pcl::copyPointCloud(*organizedCloud, *inputptr);
	cout << "samplePointsFromMesh end size: " << inputptr->size() << endl;
}

//Determine if points in grid point cloud are missing
bool checkForMissingPoints(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr) {
	cout << "checkForMissingPoints start" << endl;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr buffer(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PassThrough<pcl::PointXYZRGBA> pass;
	pass.setInputCloud(inputptr);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.99, 1.01);
	pass.filter(*buffer);
	if (buffer->size() > 0) {
		cout << "checkForMissingPoints: Nbr of missing points: " << buffer->size() << endl;
		return true;
	}
	else {
		cout << "checkForMissingPoints: Nbr of missing points: " << buffer->size() << endl;
		return false;
	}
	cout << "checkForMissingPoints end" << endl;
}

//Interpolate missing z value based on surrounding values
double interpolateZvalueBasedOnSurroundings(double p0, double p1, double p2, double p3) {
	bool linear = true;
	double result = 0;
	if (linear) {
		double Xbar = 1.5;
		double Ybar = (p0 + p1 + p2 + p3) / 4;
		double m = (((0 - 1.5) * (p0 - Ybar)) + ((1 - 1.5) * (p1 - Ybar)) + ((2 - 1.5) * (p2 - Ybar)) + ((3 - 1.5) * (p3 - Ybar)))
			/ (pow(0 - 1.5, 2) + pow(1 - 1.5, 2) + pow(2 - 1.5, 2) + pow(3 - 1.5, 2));
		double b = Ybar - m * Xbar;
		result = m * 4 + b;
		if (isnan(result) || isinf(result)) {
			return 1;
		}
		else if (result > 1 || result < -1) {
			return result;
		}
		else {
			return result;
		}
	}
	else {
		double d1 = 0.5 * (p2 - p0);
		double d2 = 0.5 * (p3 - p1);
		double a0 = p1;
		double a1 = d1;
		double a2 = (3.0 * (p2 - p1)) - (2.0 * d1) - d2;
		double a3 = d1 + d2 + (2.0 * (-p2 + p1));
		double t = 3.0;
		result = a0 + a1 * t + a2 * t * t + a3 * t * t * t;
		if (isnan(result) || isinf(result)) {
			return 1;
		}
		else if (result > 1 || result < -1) {
			return result;
		}
		else {
			return result;
		}
	}
}

//determines if any of the four points are missing. (their value would be 1)
bool checkIfNoNeighborsAreMissing(double p0, double p1, double p2, double p3) {
	if (p0 == 1 || p1 == 1 || p2 == 1 || p3 == 1) {
		return false;
	}
	else {
		return true;
	}
}

//determine the points priority based on the proximity to missing points are distance to border. Fix highest priority points first. 
void determinePriorities(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, int iy, int ix, std::vector<double>& gridParameters, std::vector<int>& priorities)
{
	int xDistBorder = gridParameters[2] - ix;
	if (xDistBorder > ix) {
		xDistBorder = ix;
	}
	int yDistBorder = gridParameters[2] - iy;
	if (yDistBorder > iy) {
		yDistBorder = iy;
	}
	//top
	if (iy > 3) {
		if (checkIfNoNeighborsAreMissing(inputptr->at(ix, iy - 4).z, inputptr->at(ix, iy - 3).z, inputptr->at(ix, iy - 2).z, inputptr->at(ix, iy - 1).z)) {

			if (yDistBorder < xDistBorder) {
				priorities.push_back(2);
			}
			else {
				priorities.push_back(1);
			}
		}
		else {
			priorities.push_back(0);
		}
	}

	else {
		priorities.push_back(-1);
	}
	//right side
	if (ix < (gridParameters[2] - 4)) {
		if (checkIfNoNeighborsAreMissing(inputptr->at(ix + 4, iy).z, inputptr->at(ix + 3, iy).z, inputptr->at(ix + 2, iy).z, inputptr->at(ix + 1, iy).z)) {
			if (yDistBorder > xDistBorder) {
				priorities.push_back(2);
			}
			else {
				priorities.push_back(1);
			}
		}
		else {
			priorities.push_back(0);
		}
	}
	else {
		priorities.push_back(-1);
	}
	//bottom
	if (iy < (gridParameters[3] - 4)) {
		if (checkIfNoNeighborsAreMissing(inputptr->at(ix, iy + 4).z, inputptr->at(ix, iy + 3).z, inputptr->at(ix, iy + 2).z, inputptr->at(ix, iy + 1).z)) {
			if (yDistBorder < xDistBorder) {
				priorities.push_back(2);
			}
			else {
				priorities.push_back(1);
			}
		}
		else {
			priorities.push_back(0);
		}
	}
	else {
		priorities.push_back(-1);
	}
	//left side
	if (ix > 3) {
		if (checkIfNoNeighborsAreMissing(inputptr->at(ix - 4, iy).z, inputptr->at(ix - 3, iy).z, inputptr->at(ix - 2, iy).z, inputptr->at(ix - 1, iy).z)) {
			if (yDistBorder > xDistBorder) {
				priorities.push_back(2);
			}
			else {
				priorities.push_back(1);
			}
		}
		else {
			priorities.push_back(0);
		}
	}
	else {
		priorities.push_back(-1);
	}
}

double checkProximityPoint(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, std::vector<double>& gridParameters, int ix, int iy) {
	std::vector<int> priorities;
	determinePriorities(inputptr, iy, ix, gridParameters, priorities);
	//check highest priority
	int maxPriority = *max_element(std::begin(priorities), std::end(priorities));
	double result = 0;
	int ctr = 0;
	if (maxPriority > 0) {
		for (int i = 0; i < 4; i++) {
			if (priorities[i] == maxPriority) {
				if (i == 0) {

					result = result + (interpolateZvalueBasedOnSurroundings(inputptr->at(ix, iy - 4).z, inputptr->at(ix, iy - 3).z, inputptr->at(ix, iy - 2).z, inputptr->at(ix, iy - 1).z));
					ctr++;
				}
				else if (i == 1) {
					result = result + (interpolateZvalueBasedOnSurroundings(inputptr->at(ix + 4, iy).z, inputptr->at(ix + 3, iy).z, inputptr->at(ix + 2, iy).z, inputptr->at(ix + 1, iy).z));
					ctr++;
				}
				else if (i == 2) {
					result = result + (interpolateZvalueBasedOnSurroundings(inputptr->at(ix, iy + 4).z, inputptr->at(ix, iy + 3).z, inputptr->at(ix, iy + 2).z, inputptr->at(ix, iy + 1).z));
					ctr++;
				}
				else if (i == 3) {
					result = result + (interpolateZvalueBasedOnSurroundings(inputptr->at(ix - 4, iy).z, inputptr->at(ix - 3, iy).z, inputptr->at(ix - 2, iy).z, inputptr->at(ix - 1, iy).z));
					ctr++;
				}
			}
		}
		result = result / ctr;
		return result;
	}
	else {
		return 1;
	}
}

//Order of points to move in a spiral through grid. This way holes can be filled from the inside out.
//Improves hole filling
void spiralOrder(std::vector<std::vector<int>>& order, int R, int C)
{
	std::vector<std::vector<bool> > seen(R, std::vector<bool>(C, false));
	int dr[] = { 0, 1, 0, -1 };
	int dc[] = { 1, 0, -1, 0 };
	int r = 0, c = 0, di = 0;
	// Iterate from 0 to R * C - 1
	for (int i = 0; i < R * C; i++) {
		std::vector<int> bufferPair;
		bufferPair.push_back(r);
		bufferPair.push_back(c);
		order.push_back(bufferPair); //ans.push_back(matrix[r][c]);
		seen[r][c] = true;
		int cr = r + dr[di];
		int cc = c + dc[di];
		if (0 <= cr && cr < R && 0 <= cc && cc < C
			&& !seen[cr][cc]) {
			r = cr;
			c = cc;
		}
		else {
			di = (di + 1) % 4;
			r += dr[di];
			c += dc[di];
		}
	}
}

//Fills holes in point cloud after grid creation
void fillHolesInGrid(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, std::vector<double>& gridParameters) {
	cout << "fillHolesInGrid start  " << endl;
	//Search smallest rectangle after with seeds on the outside. Crop point cloud the fill. 
	int ctr = 0;
	std::vector<std::vector<int>> order;
	spiralOrder(order, gridParameters[2], gridParameters[3]);
	while (checkForMissingPoints(inputptr)) {
		ctr++;
		for (int i = order.size() - 1; i >= 0; i--) {
			if (inputptr->at(order[i][0], order[i][1]).z == 1) {
				inputptr->at(order[i][0], order[i][1]).z = checkProximityPoint(inputptr, gridParameters, order[i][0], order[i][1]);
			}
		}
	}
	cout << "fillHolesInGrid end  " << endl;
}

//Crops the point cloud to a rectangle.
void cropPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, std::vector<double>& gridParameters) {
	cout << "Start  cropPointCloud" << endl;
	std::vector<std::vector<int>> missingDataMatrix; //0 not missing // 1 missing // 2 missing and connected to the sides.
	for (int ix = 0; ix < gridParameters[2]; ix++) {
		std::vector<int > missingDataVector;
		for (int iy = 0; iy < gridParameters[3]; iy++) {
			if (inputptr->at(ix, iy).z == 1) {
				missingDataVector.push_back(1);
			}
			else {
				missingDataVector.push_back(0);
			}
		}
		missingDataMatrix.push_back(missingDataVector);
	}
	std::vector<std::vector<int>> order;
	//Spiral order from outside to inside
	spiralOrder(order, gridParameters[2], gridParameters[3]);
	for (int i = 0; i < order.size(); i++) {
		if (missingDataMatrix[order[i][0]][order[i][1]] == 1) {
			if (order[i][0] == 0 || order[i][0] == gridParameters[2] - 1 || order[i][1] == 0 || order[i][1] == gridParameters[3] - 1) {
				missingDataMatrix[order[i][0]][order[i][1]] = 2;
			}
			else if (missingDataMatrix[order[i][0] - 1][order[i][1]] == 2 || missingDataMatrix[order[i][0] + 1][order[i][1]] == 2 ||
				missingDataMatrix[order[i][0]][order[i][1] - 1] == 2 || missingDataMatrix[order[i][0]][order[i][1] + 1] == 2) {
				missingDataMatrix[order[i][0]][order[i][1]] = 2;
			}
		}
	}
	bool missingCornerPointsNotDeleted = true;
	int xBegin = 0;
	int xEnd = gridParameters[2];
	int yBegin = 0;
	int yEnd = gridParameters[3];
	//TODO Create more comments
	while (missingCornerPointsNotDeleted) {
		//top
		missingCornerPointsNotDeleted = false;
		for (int x = xBegin; x < xEnd; x++) {
			if (missingDataMatrix[x][yBegin] == 2) {
				yBegin++;
				missingCornerPointsNotDeleted = true;
				break;
			}
		}
		//right
		for (int y = yBegin; y < yEnd; y++) {
			if (missingDataMatrix[xEnd - 1][y] == 2) {
				xEnd--;
				missingCornerPointsNotDeleted = true;
				break;
			}
		}
		//bottom
		for (int x = xBegin; x < xEnd; x++) {
			if (missingDataMatrix[x][yEnd - 1] == 2) {
				yEnd--;
				missingCornerPointsNotDeleted = true;
				break;
			}
		}
		//left
		for (int y = yBegin; y < yEnd; y++) {
			if (missingDataMatrix[xBegin][y] == 2) {
				xBegin++;
				missingCornerPointsNotDeleted = true;
				break;
			}
		}
	}
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr organizedCloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
	organizedCloud->width = xEnd - xBegin;
	organizedCloud->height = yEnd - yBegin;
	organizedCloud->is_dense = false;
	organizedCloud->points.resize(organizedCloud->height * organizedCloud->width);
	for (int ix = xBegin; ix < xEnd; ix++) {
		for (int iy = yBegin; iy < yEnd; iy++) {
			organizedCloud->at(ix - xBegin, iy - yBegin).x = inputptr->at(ix, iy).x;
			organizedCloud->at(ix - xBegin, iy - yBegin).y = inputptr->at(ix, iy).y;
			organizedCloud->at(ix - xBegin, iy - yBegin).z = inputptr->at(ix, iy).z;
		}
	}
	pcl::copyPointCloud(*organizedCloud, *inputptr);
	gridParameters[2] = xEnd - xBegin;
	gridParameters[3] = yEnd - yBegin;
	cout << "cropPointCloud end" << endl;
}

void createDenseGridPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampleValue, boost::shared_ptr< ::pcl::PolygonMesh>  meshPCL, bool stlInput) {
	cout << "createDenseGridPointCloud Start  " << endl;
	Polyhedron mesh;
	if (stlInput == false) {
		formRemovalPlane(inputptr);
		mesh = denseMesh(inputptr);

	}
	else {
		std::vector<pcl::Vertices> mesh_vertices;
		mesh_vertices = meshPCL->polygons;
		std::vector< CGAL::cpp11::array<double, 3> > points2;
		std::vector< CGAL::cpp11::array<int, 3> > triangles;
		for (int g = 0; g < inputptr->size(); g++) {
			points2.push_back({ inputptr->points[g].x, inputptr->points[g].y, inputptr->points[g].z });
		}
		for (int g = 0; g < mesh_vertices.size(); g++) {
			triangles.push_back({ (int)mesh_vertices[g].vertices[0],(int)mesh_vertices[g].vertices[1] ,(int)mesh_vertices[g].vertices[2] });
		}
		CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points2, triangles, mesh);
		if (saveClouds) {
			std::stringstream ss;;
			ss << dir << file << "StlGrid.off";
			std::ofstream f(ss.str().c_str());
			f << mesh;
			f.close();
		}
	}
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr gridPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::copyPointCloud(*inputptr, *gridPointCloud);
	std::vector<double> gridParameters;
	samplePointsFromMesh(gridPointCloud, mesh, downsampleValue, gridParameters);
	cropPointCloud(gridPointCloud, gridParameters);
	fillHolesInGrid(gridPointCloud, gridParameters);
	pcl::copyPointCloud(*gridPointCloud, *inputptr);
	cout << "createDenseGridPointCloud end  " << endl;
}

//Compare to ISO 16610-71 (8)
double gaussianWeightingFunction(double cutoffWavelength, double xKI, double yLJ) {
	return ((1 / (pow(0.7309, 2) * pow(cutoffWavelength, 2))) * exp((-M_PI / pow(0.7309, 2)) * ((pow(xKI, 2) + pow(yLJ, 2)) / pow(cutoffWavelength, 2))));
}

//determines the kernel sizes, the kernels themselves and applies them to the input point cloud. 
void filterShortLong(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsample, bool cropCloud, string dir, string file, float cutoffWaveLengthLow, float cutoffWaveLengthHigh) {
	cout << "filterSortLong Start " << "inputptr->size(): " << inputptr->size() << endl;
	int KernelSizeLow = determineKernelSizeFromCutOff(cutoffWaveLengthLow, downsample);
	int KernelSizeHigh = determineKernelSizeFromCutOff(cutoffWaveLengthHigh, downsample);
	MatrixXf GKernelLow(KernelSizeLow, KernelSizeLow);
	MatrixXf GKernelHigh(KernelSizeHigh, KernelSizeHigh);
	GKernelLow = crateGaussFilterKernel(GKernelLow, cutoffWaveLengthLow, KernelSizeLow, false);
	GKernelHigh = crateGaussFilterKernel(GKernelHigh, cutoffWaveLengthHigh, KernelSizeHigh, true);
	applyGaussianFilterToCloud(inputptr, GKernelLow, KernelSizeLow, false, cropCloud, dir, file);
	applyGaussianFilterToCloud(inputptr, GKernelHigh, KernelSizeHigh, true, cropCloud, dir, file);
	cout << " applyFilterCloud end: cloud->height: " << inputptr->height << " cloud->width: " << inputptr->width << endl;
}

//3D arithmetric mean
double Sa(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr) {
	double sum = 0.0;
	for (int i = 0; i < inputptr->size(); i++) {
		sum = sum + abs(inputptr->points[i].z);
	}
	double sa = sum / inputptr->size();
	return sa;
}

//3D square root
double Sq(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr) {
	double sum = 0.0;
	for (int i = 0; i < inputptr->size(); i++) {
		sum = sum + pow(inputptr->points[i].z, 2);
	}
	double sq = sqrt(sum / inputptr->size());
	return sq;
}

//Create rectangle and crop outside points
std::vector<MatrixXf> CreateEvaluationLengthMatrix(double downsampling, int PointsOnVariogram, double span) {
	cout << "CreateEvaluationLengthMatrix start  " << endl;
	int maxGridDistance = ceil(span * PointsOnVariogram / downsampling);
	float maxEv = PointsOnVariogram * span;
	int matrixSize = maxGridDistance * 2 + 1;
	MatrixXf EvalMatrix(matrixSize, matrixSize);
	MatrixXf DistMatrix(matrixSize, matrixSize);
	for (int i = 0; i < matrixSize; i++) {
		for (int m = 0; m < matrixSize; m++) {
			double dist = sqrt(pow((i - maxGridDistance) * downsampling, 2) + pow((m - maxGridDistance) * downsampling, 2));
			DistMatrix(i, m) = dist;
			if (dist > maxEv || (i == maxGridDistance && m == maxGridDistance)) {
				EvalMatrix(i, m) = -1;
			}
			else {
				EvalMatrix(i, m) = floor(dist / span);
			}
		}
	}
	std::vector<MatrixXf> EMatrix;
	EMatrix.push_back(EvalMatrix);
	EMatrix.push_back(DistMatrix);
	cout << "CreateEvaluationLengthMatrix end  " << endl;
	return EMatrix;
}

//calculate Variogram roughness
double Svr(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampling, int PointsOnVariogram, double span, std::vector<double>& SvrGauss) {
	cout << "Svr start" << endl;
	std::vector < MatrixXf> EvalMatrix = CreateEvaluationLengthMatrix(downsampling, PointsOnVariogram, span);
	MatrixXf EMatrix = EvalMatrix[0];
	int EvalMSize = EMatrix.rows() / 2.0 - 0.5;
	int height = inputptr->height;
	int width = inputptr->width;
	//Create storage Structure
	std::vector < std::vector < std::vector <double >>> SumMatrix;
	std::vector < std::vector < std::vector <double >>>  CtrMatrix;
	MatrixXf SvrMatrix(inputptr->height, inputptr->width);
	for (int i = 0; i < PointsOnVariogram + 1; i++) {
		std::vector < std::vector <double >> SumMatrix2;
		std::vector < std::vector <double >> CtrMatrix2;

		for (int ii = 0; ii < inputptr->width; ii++) {
			std::vector <double> SumMatrix3;
			std::vector <double> CtrMatrix3;
			for (int iii = 0; iii < inputptr->height; iii++) {
				SumMatrix3.push_back(0);
				CtrMatrix3.push_back(0);
			}
			SumMatrix2.push_back(SumMatrix3);
			CtrMatrix2.push_back(CtrMatrix3);
		}
		SumMatrix.push_back(SumMatrix2);
		CtrMatrix.push_back(CtrMatrix2);
	}
	//Determine sum and counter
	size_t size = inputptr->width;
	cout << "Svr parallel for start" << endl;
	parallel_for(size_t(0), size, [&](size_t i) {
		//for (int i = 0; i < inputptr->width; i++) {
		for (int m = 0; m < inputptr->height; m++) {
			int ix = i;
			//int mx = m;
			for (int kk = 0; kk < PointsOnVariogram; kk++) {
				SumMatrix[kk][i][m] = 0;
				CtrMatrix[kk][i][m] = 0;
			}
			int minXGrid = 0, maxXGrid = EMatrix.rows(), minYGrid = 0, maxYGrid = EMatrix.rows();
			if (ix - EvalMSize < 0) {
				minXGrid = abs(ix - EvalMSize);
			}
			else if (abs(ix - (width - 1)) < EvalMSize) {
				int distToEdge = abs(ix - (width - 1));
				maxXGrid = (EMatrix.rows() - (EvalMSize - distToEdge));
			}
			if (m - EvalMSize < 0) {
				minYGrid = abs(m - EvalMSize);
			}
			else if (abs(m - (height - 1)) < EvalMSize) {
				int distToEdge = abs(m - (height - 1));
				maxYGrid = (EMatrix.rows() - (EvalMSize - distToEdge));
			}
			for (int x = minXGrid; x < maxXGrid - 1; x++) {
				for (int y = minYGrid; y < maxYGrid - 1; y++) {
					int k = EMatrix(x, y);
					if (k != -1) {
						int ii = i - (EvalMSize - x);
						int mm = m - (EvalMSize - y);
						//int k2 = k;
						k = floor(distance(inputptr->at(i, m), inputptr->at(ii, mm)) / span);
						if (k > -1 && k < PointsOnVariogram) {
							SumMatrix[k][i][m] += pow(1000 * (inputptr->at(i, m).z - inputptr->at(ii, mm).z), 2);
							CtrMatrix[k][i][m] += 1;
						}
					}
				}
			}
		}
		//}
		});
	cout << "Svr parallel for end" << endl;
	std::vector<float> sumVec(PointsOnVariogram, 1);
	std::vector<float> CtrVec(PointsOnVariogram, 1);
	std::vector<double> varVec(PointsOnVariogram, 1);
	double Svr3 = 0;
	double CtrAll = 0;
	//calculate roughness from sum and counter.
	for (int k = 0; k < PointsOnVariogram; k++) {
		sumVec[k] = 0;
		CtrVec[k] = 0;
		varVec[k] = 0;
		for (int i = 0; i < inputptr->width; i++) {
			for (int m = 0; m < inputptr->height; m++) {
				sumVec[k] += SumMatrix[k][i][m];  // accumulate(sumVecParallel[i].begin(), sumVecParallel[i].end(), 0);
				CtrVec[k] += CtrMatrix[k][i][m];  // accumulate(CtrVecParallel[i].begin(), CtrVecParallel[i].end(), 0);
			}
		}
		if (CtrVec[k] > 0) {
			varVec[k] = sqrt((1 / (2 * CtrVec[k])) * sumVec[k]);
			CtrAll += CtrVec[k];
			Svr3 += varVec[k];
		}
		else {
			cout << "Warning: no point pairs were found for: " << k << endl;
		}
	}
	SvrGauss = varVec;
	Svr3 = Svr3 / PointsOnVariogram;
	cout << "Svr end" << endl;
	return Svr3;
}

//This method calculates the roughness parameters Sa, Sq and Svr for a gauss filtered point cloud grid. 
void roughnessCalculation(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& inputptr, double downsampling, int PointsOnVariogram, double span,
	std::vector<double>& SvrGauss, std::vector<double>& SaSqSvrGauss, int RoughnessParameterSaSqSvr, bool OverWriteParameters) {
	cout << "roughnessCalculation Start" << endl;
	double sa = 0;
	double sq = 0;
	double svr = 0;
	if (RoughnessParameterSaSqSvr == 0 || OverWriteParameters == true) {
		sa = Sa(inputptr) * 1000000;
	}
	if (RoughnessParameterSaSqSvr == 1 || OverWriteParameters == true) {
		sq = Sq(inputptr) * 1000000;
	}
	if (RoughnessParameterSaSqSvr == 2 || OverWriteParameters == true) {
		svr = Svr(inputptr, downsampling, PointsOnVariogram, span, SvrGauss) * 1000;
	}
	SaSqSvrGauss.push_back(sa);
	SaSqSvrGauss.push_back(sq);
	SaSqSvrGauss.push_back(svr);

	cout << "roughnessCalculation End" << endl;
}

double calculateSurfaceRoughnessOfPointCloud(std::string filepath) {
	namespace fs = std::filesystem;
	const fs::path inputPath(filepath);
	if (!fs::exists(inputPath)) {
		std::cerr << "Input file does not exist: " << filepath << std::endl;
		return -1;
	}

	// These mirror the current SurfInspect GUI run:
	// downsample=0.3 mm, mesh=1 mm, span=0.5 mm, evaluation length=5 mm,
	// target edge=5 mm, PointsOnVariogram=10, global Gaussian mesh, and
	// 1-25 mm wavelength cutoffs.
	const bool unitMM = true;
	const bool unitInch = false;
	const bool statFilter = false;
	const bool gridFilter = false;
	const bool removeEdgesAfterMeshing = false;
	const bool globalMeshing = true;
	const bool bezier = false;
	const bool gaussDistance = false;
	const bool gaussGrid = true;
	const bool overwriteParameters = false;
	const int pointsOnVariogram = 10;
	const int roughnessParameter = 2;
	const double downsample = 0.0003;
	const double meshVoxelSize = 0.001;
	const double span = 0.0005;
	const double targetEdgeLength = 0.005;
	const double longWavelengthCutoff = 0.025;
	const double shortWavelengthCutoff = 0.001;

	std::string filename = inputPath.filename().string();
	std::string fileEnding = inputPath.extension().string();
	std::string file = inputPath.stem().string();
	std::string outputDir = inputPath.parent_path().string();
	if (!outputDir.empty() && outputDir.back() != '\\' && outputDir.back() != '/') {
		outputDir += "\\";
	}
	std::string diagnosticDir = outputDir + file + "_legacy_diagnostics";
	fs::create_directories(diagnosticDir);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBACombined(new pcl::PointCloud<pcl::PointXYZRGBA>);
	boost::shared_ptr<pcl::PolygonMesh> inputMesh(new pcl::PolygonMesh);
	bool stlInput = false;
	stlInput = LoadInputFromFile(fileEnding, filepath, cloud, inputMesh, stlInput);
	if (cloud->empty()) {
		std::cerr << "Could not load input file: " << filepath << std::endl;
		return -1;
	}
	const size_t loadedPointCount = cloud->size();
	std::cout << "[legacy] loaded: " << loadedPointCount << std::endl;

	pcl::PointXYZRGBA centroid;
	CenterPointCloudAndAdjustUnits(cloud, centroid, unitMM, unitInch);
	pcl::io::savePCDFileBinary(diagnosticDir + "\\01_centered.pcd", *cloud);
	std::cout << "[legacy] centered: " << cloud->size() << std::endl;

	VoxelGridDownsample(cloud, downsample);
	pcl::io::savePCDFileBinary(diagnosticDir + "\\02_initial_voxel.pcd", *cloud);
	std::cout << "[legacy] initial_voxel: " << cloud->size() << std::endl;

	std::vector<double> variogram(pointsOnVariogram, 0.0);
	std::vector<std::vector<double>> variogramMatrix;
	std::vector<double> svrPoints;
	std::vector<double> distancePoints;
	std::vector<double> saSqSvr;
	bool retflag = false;
	auto begin = std::chrono::steady_clock::now();
	const int result = prepareVariogramCalculation(
		cloud, cloudRGBACombined, outputDir, file, statFilter, gridFilter,
		removeEdgesAfterMeshing, downsample, meshVoxelSize, centroid,
		globalMeshing, bezier, targetEdgeLength, begin, variogram,
		pointsOnVariogram, span, variogramMatrix, svrPoints, distancePoints,
		retflag, gaussDistance, gaussGrid, saSqSvr, roughnessParameter,
		overwriteParameters, longWavelengthCutoff, shortWavelengthCutoff);
	if (result != 0 || retflag || saSqSvr.size() <= static_cast<size_t>(roughnessParameter)) {
		std::cerr << "Legacy SurfInspect calculation failed." << std::endl;
		return -1;
	}

	std::ofstream summary(diagnosticDir + "\\summary.txt");
	summary << "input=" << filename << "\n";
	summary << "loaded=" << loadedPointCount << "\n";
	summary << "initial_voxel=" << cloud->size() << "\n";
	summary << "downsample=" << downsample << "\n";
	summary << "mesh_voxel=" << meshVoxelSize << "\n";
	summary << "points_on_variogram=" << pointsOnVariogram << "\n";
	summary << "span=" << span << "\n";
	summary << "svr_um=" << saSqSvr[roughnessParameter] << "\n";
	summary.close();

	std::cout << "[legacy] svr_um: " << saSqSvr[roughnessParameter] << std::endl;
	std::cout << "[legacy] diagnostics: " << diagnosticDir << std::endl;
	return saSqSvr[roughnessParameter];
}

//Used in main
//This is the primary function that performs all of Daniel's surface roughness code. General structure is: 
//1. Point cloud aquisition either by opening a stored file or initiating a scan with a connected sensor. 
//2. Preparation of the point cloud such as downsampling and applying statistical filters.
//3. Determining the underlying geometry. This involces further downsampling and then matching a mesh to the points
//4. Determining the distances between the point cloud and the mesh
//5. Calculating the variogram values and coloring the point cloud
//int calculateSurfaceRoughnessOfPointCloud(std::string filepath) {
//	//Save start time
//	std::chrono::steady_clock::time_point beginMain = std::chrono::steady_clock::now();
//
//	// Startup Variables
//	std::string filename = filepath;
//	bool traditional, UnitMM, UnitInch, GaussDistance, gaussGrd;
//	bool load = true;
//
//	double downsample = 0.0002;
//	double cell_size_grid_simplify_point_set = 0.001;
//	int PointsOnVariogram = 10;
//	double span = 0.0005;
//	double target_edge_length = 0.005;
//	string startArgString = "%#StartArgs#%";
//	bool GridFilterBool = true;
//	bool removeEdgesAfterMeshing = false;
//	bool removeEdges = false;
//	bool statFilter = true;
//	bool globalMeshing = true;
//	bool detectAbnormalities = false;
//	bool bezier = false;
//	int abnormalitySelection = 0; //0 all, 1 depression, 2 hills
//	bool removeAbnormalitiesAndRepeat = false;
//	bool ShowPointCloud = false;
//	bool cropCloud = false;
//	saveClouds = true;
//	//float cutoffWaveLengthLow = 0.001;
//	//float cutoffWaveLengthHigh = 0.025;
//
//	int unit = 0;  //mm, m, inch
//	int underlyingGeometry = 0;  //
//	double shortWavelengthCutoff = 0.001;
//	double longWavelengthCutoff = 0.025;
//	int RoughnessParameterSaSqSvr = 2; //Sa, Sq, Svr
//	int InputType = 0; // Section, Box
//	bool OverWriteParameters = false;
//	float evaluationLength = 0.005;
//	float variogramPointSpacing = 0.0005;
//
//	//convert dropdown menu values to variables used here.
//	if (unit == 0) {
//		UnitMM = true;
//		UnitInch = false;
//	}
//	else if (unit == 1) {
//		UnitMM = false;
//		UnitInch = false;
//	}
//	else if (unit == 2) {
//		UnitMM = false;
//		UnitInch = true;
//	}
//	if (underlyingGeometry == 0) {
//		traditional = false;
//		GaussDistance = false;
//		gaussGrd = false;
//	}
//	else if (underlyingGeometry == 1) {
//		traditional = true;
//		GaussDistance = true;
//		gaussGrd = false;
//	}
//	else if (underlyingGeometry == 2) {
//		traditional = true;
//		GaussDistance = false;
//		gaussGrd = true;
//	}
//	else if (underlyingGeometry == 3) {
//		traditional = true;
//		GaussDistance = false;
//		gaussGrd = false;
//	}
//	cout << "Operator: " << Operator << " Casting: " << Casting << " Run: " << Run << " GageRR: " << GageRR << endl;
//	cout << "GaussDistance: " << GaussDistance << " gaussGrd: " << gaussGrd << " cropCloud: " << cropCloud << endl;
//	cout << "downsample" << ": " << downsample << endl;
//	cout << "unit" << ": " << unit << endl;
//	cout << "underlyingGeometry" << ": " << underlyingGeometry << endl;
//	cout << "shortWavelengthCutoff" << ": " << shortWavelengthCutoff << endl;
//	cout << "longWavelengthCutoff" << ": " << longWavelengthCutoff << endl;
//	cout << "cropCloud" << ": " << cropCloud << endl;
//	cout << "globalMeshing" << ": " << globalMeshing << endl;
//	cout << "removeEdges" << ": " << removeEdges << endl;
//	cout << "traditional" << ": " << traditional << endl;
//
//	std::size_t botDirPos = filename.find_last_of("\\");
//	dir = filename.substr(0, botDirPos + 1);
//	file = filename.substr(botDirPos + 1, filename.length());
//
//	botDirPos = file.find_last_of(".");
//	std::string fileEnding = file.substr(botDirPos, file.length());
//	file = file.substr(0, botDirPos);
//	cout << "Filename: " << file << endl;
//	cout << "Directory: " << dir << endl;
//	cout << "fileEnding: " << fileEnding << endl;
//	std::cout << "Save cloud as: " << filename << std::endl << std::endl;
//
//
//
//
//	//2. Make this a function: loadPointCloud
//	//Create empty point cloud pointers
//	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudRGBACombined(new pcl::PointCloud<pcl::PointXYZRGBA>);
//	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
//	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGBA>);
//	//pcl::PointCloud < pcl::PolygonMesh>::Ptr inputMesh(new pcl::PointCloud<pcl::PolygonMesh>);
//	boost::shared_ptr< ::pcl::PolygonMesh> inputMesh(new pcl::PolygonMesh);
//	bool stlInput = false;
//	//std::vector<double> varVec(PointsOnVariogram, 0);
//	std::vector <std::vector<double>> varMatrix;
//	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//
//	//load Point cloud file
//	if (load) {
//		stlInput = LoadInputFromFile(fileEnding, filename, cloud, inputMesh, stlInput);
//	}
//	else {
//		pcl::copyPointCloud(*cloudGLOBAL, *cloud);
//	}
//	if (GageRR) {
//		std::stringstream buffss;
//		buffss << "C:\\Users\\dschimpf\\Box\\Daniel Schimpf\\RE\\Surface Project\\PCL_Project\\GageRR\\Gage_Original_O" << Operator << "_C" << Casting << "_R" << Run << ".pcd";
//		pcl::io::savePCDFileASCII(buffss.str(), *cloud);
//	}
//	//First load the pcl mesh, then manipulate the point cloud, move to center, change unit. 
//	//this should not change connection between points. 
//	//change to cgal mesh in createDenseGridPOinctCloud
//	//exclude not acceptable operations.
//
//	pcl::PointXYZRGBA outCentroid2;
//	CenterPointCloudAndAdjustUnits(cloud, outCentroid2, UnitMM, UnitInch);
//
//
//	//Make this a function: 
//	if (statFilter && stlInput == false) {
//		StatOutlierRemoval(cloud);
//		if (saveClouds) {
//			pcl::io::savePCDFileASCII(dir + file + "Statfiltered.pcd", *cloud);
//		}
//		cout << "Stat filter saved %%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;
//	}
//	pcl::copyPointCloud(*cloud, *cloud2);
//	std::vector<double> SvrVec(PointsOnVariogram, 0);
//	std::vector<double> SaSqSvr;
//	double SvrFinal = 0;
//	// traditional in the sense of what Daniel started with. That was a grid based underlying geometry detection. 
//	//non traditional is the gaussian z value adjustment. 
//	if (traditional == false) {
//		cout << "nontraditional start   stlInput: " << stlInput << endl;
//		if (stlInput == false) {
//			VoxelGridDownsample(cloud, downsample);
//		}
//		createDenseGridPointCloud(cloud, downsample, inputMesh, stlInput);
//		filterShortLong(cloud, downsample, cropCloud, dir, file, shortWavelengthCutoff, longWavelengthCutoff);
//		roughnessCalculation(cloud, downsample, PointsOnVariogram, span, SvrVec, SaSqSvr, RoughnessParameterSaSqSvr, OverWriteParameters);
//		cout << "nontraditional end" << endl;
//	}
//	else {
//		cout << "traditional start" << endl;
//		pcl::copyPointCloud(*cloud2, *cloud);
//		//downsamples based on voxels in space
//		VoxelGridDownsample(cloud, downsample);
//
//		//This searches for edges
//		std::vector<pcl::PointIndices> cluster_indices;
//		if (removeEdges) {
//			DetermineEdgesAndCluster(cloud, dir, cloud, removeEdgesAfterMeshing, cluster_indices);
//		}
//		std::cout << "Time for edge detection = " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginMain).count() << "[s]" << std::endl;
//		std::vector<double> SvrPoint;
//		std::vector<double> SvrPointCombined;
//		std::vector<double> dist2Combined;
//		//std::vector<double> SaSqSvr;
//
//		//InputType 0 means that the input is a surface section. 1 means complex 3D data is used as an input.
//		if (InputType == 0) {
//			cout << "Surface Section start" << endl;
//			bool retflag;
//			if (-1 == prepareVariogramCalculation(cloud, cloudRGBACombined, dir, file, statFilter, GridFilterBool, removeEdgesAfterMeshing,
//				downsample, cell_size_grid_simplify_point_set, outCentroid2, globalMeshing, bezier, target_edge_length,
//				begin, /*Svr, */SvrVec, PointsOnVariogram, span, varMatrix, SvrPointCombined, dist2Combined, retflag, GaussDistance, gaussGrd, SaSqSvr,
//				RoughnessParameterSaSqSvr, OverWriteParameters, longWavelengthCutoff, shortWavelengthCutoff)) {
//				return -1;
//			}
//
//			float calculationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - beginMain).count();
//			std::cout << "Main Function Execution Time: " << calculationTime << "[ms]" << std::endl;
//			cout << "Sa: " << SaSqSvr[0] << endl;
//			cout << "Sq: " << SaSqSvr[1] << endl;
//			cout << "Svr: " << SaSqSvr[2] << endl;
//			cout << "SvrFinal: " << SvrFinal << endl;
//			if (globalMeshing == true && detectAbnormalities == true) {
//				colorPcBasedOnAbnormalities(SvrPointCombined, cloudRGBACombined, SvrFinal, dist2Combined, abnormalitySelection);
//				if (saveClouds) {
//					pcl::io::savePCDFileASCII(dir + file + "Abnormalities.pcd", *cloudRGBACombined);
//				}
//			}
//
//			if (removeAbnormalitiesAndRepeat) {
//				pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
//				pcl::PointIndices::Ptr inliers2(new pcl::PointIndices());
//				for (int i = 0; i < cloudRGBACombined->size(); i++) {
//					if (cloudRGBACombined->points[i].g == 255) {
//						inliers2->indices.push_back(i);
//					}
//				}
//				extract.setInputCloud(cloudRGBACombined);
//				extract.setIndices(inliers2);
//				extract.setNegative(false);
//				extract.filter(*cloudRGBACombined);
//				//if (saveClouds) {
//				//	pcl::io::savePCDFileASCII(dir + file + "_AR_Colored.pcd", *cloudRGBACombined);
//				//}
//				pcl::copyPointCloud(*cloudRGBACombined, *cloud);
//				cluster_indices.clear();
//				clusterPc(cloud, cluster_indices);
//				file = file + "_AR_";
//				SvrPoint.clear();
//				SvrPointCombined.clear();
//				dist2Combined.clear();
//				std::fill(SvrVec.begin(), SvrVec.end(), 0);
//
//				varMatrix.clear();
//				cloudRGBACombined->clear();
//				int retval = prepareVariogramCalculation(cloud, cloudRGBACombined, dir, file, statFilter, GridFilterBool, removeEdgesAfterMeshing,
//					downsample, cell_size_grid_simplify_point_set, outCentroid2, globalMeshing, bezier, target_edge_length,
//					begin, /*Svr, */SvrVec, PointsOnVariogram, span, varMatrix, SvrPointCombined, dist2Combined, retflag, GaussDistance, gaussGrd, SaSqSvr,
//					RoughnessParameterSaSqSvr, OverWriteParameters, longWavelengthCutoff, shortWavelengthCutoff);
//				if (retflag) return retval;
//			}
//			cout << "Surface Section end" << endl;
//		}
//		else if (InputType == 1) {
//			cout << "Complex Section start %%%%%%%%%%%%%%%%%%%" << endl;
//			int j = 0;
//			bool retflag;
//			int retval = RoughnessCalcCluster(cluster_indices, /*j,*/ cloud, /*cloudOnSurface,*/ dir, file, statFilter, GridFilterBool, removeEdgesAfterMeshing,
//				downsample, /*cloud_buffer, */cell_size_grid_simplify_point_set, outCentroid2, globalMeshing, bezier, target_edge_length,  /*output_mesh,*/
//				begin,/* Svr,*/ SvrVec, /*sumVec, CtrVec,*/ PointsOnVariogram, span, /*sumMatrix, CtrMatrix,*/ varMatrix, /*cloudRGBA,*/ cloudRGBACombined,
//				SvrPointCombined, dist2Combined, /*buffstr, */retflag, GaussDistance, gaussGrd, SaSqSvr, RoughnessParameterSaSqSvr, OverWriteParameters,
//				longWavelengthCutoff, shortWavelengthCutoff);
//			if (retflag) return retval;
//			if (globalMeshing == true && detectAbnormalities == true) {
//				colorPcBasedOnAbnormalities(SvrPointCombined, cloudRGBACombined, SvrFinal, dist2Combined, abnormalitySelection);
//				if (saveClouds) {
//					pcl::io::savePCDFileASCII(dir + file + "Abnormalities.pcd", *cloudRGBACombined);
//				}
//			}
//			if (removeAbnormalitiesAndRepeat) {
//				pcl::ExtractIndices<pcl::PointXYZRGBA> extract;
//				pcl::PointIndices::Ptr inliers2(new pcl::PointIndices());
//				for (int i = 0; i < cloudRGBACombined->size(); i++) {
//					if (cloudRGBACombined->points[i].g == 255) {
//						inliers2->indices.push_back(i);
//					}
//				}
//				extract.setInputCloud(cloudRGBACombined);
//				extract.setIndices(inliers2);
//				extract.setNegative(false);
//				extract.filter(*cloudRGBACombined);
//				//if (saveClouds) {
//				//	pcl::io::savePCDFileASCII(dir + file + "_AR_Colored.pcd", *cloudRGBACombined);
//				//}
//				pcl::copyPointCloud(*cloudRGBACombined, *cloud);
//				cluster_indices.clear();
//				clusterPc(cloud, cluster_indices);
//				file = file + "_AR_";
//				SvrPoint.clear();
//				SvrPointCombined.clear();
//				dist2Combined.clear();
//				std::fill(SvrVec.begin(), SvrVec.end(), 0);
//				varMatrix.clear();
//				cloudRGBACombined->clear();
//				int retval = RoughnessCalcCluster(cluster_indices, /*j,*/ cloud, /*cloudOnSurface,*/ dir, file, statFilter, GridFilterBool, removeEdgesAfterMeshing,
//					downsample, /*cloud_buffer, */cell_size_grid_simplify_point_set, outCentroid2, globalMeshing, bezier, target_edge_length,  /*output_mesh,*/
//					begin,/* Svr,*/ SvrVec, /*sumVec, CtrVec,*/ PointsOnVariogram, span, /*sumMatrix, CtrMatrix,*/ varMatrix, /*cloudRGBA,*/ cloudRGBACombined,
//					SvrPointCombined, dist2Combined, /*buffstr, */retflag, GaussDistance, gaussGrd, SaSqSvr, RoughnessParameterSaSqSvr, OverWriteParameters,
//					longWavelengthCutoff, shortWavelengthCutoff);
//				if (retflag) return retval;
//			}
//			cout << "Complex Section end" << endl;
//		}
//		cout << "traditional end" << endl;
//	}
//
//	createResultTable(dir, file, varMatrix, downsample, cell_size_grid_simplify_point_set, PointsOnVariogram, span, target_edge_length, SvrVec, SvrFinal,
//		std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - beginMain).count(), cloud->size(), SaSqSvr, RoughnessParameterSaSqSvr);
//
//	//This opens up a small window in which the point cloud is visualized. TODO initial point cloud position is too small.                                                                                                                       
//	if (ShowPointCloud) {
//		Eigen::Vector4f centroid;
//		pcl::compute3DCentroid(*cloudRGBACombined, centroid);
//		Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
//		transform_2.translation() << -centroid[0], -centroid[1], -centroid[2];
//		pcl::transformPointCloud(*cloudRGBACombined, *cloudRGBACombined, transform_2);
//		pcl::visualization::PCLVisualizer::Ptr viewer = simpleVisRGBA(cloudRGBACombined);
//		std::cout << "Finished visuaizing without errors" << std::endl;
//		while (!viewer->wasStopped())
//		{
//			viewer->spinOnce(100);
//		}
//	}
//	std::cout << "DONE Total Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - beginMain).count() << "[ms]" << std::endl;
//}
