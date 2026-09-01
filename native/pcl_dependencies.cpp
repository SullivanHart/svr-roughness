#include "pcl_dependencies.h"

#include <iostream>
#include <string>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

//Remove statistical outlier. Can remove noise but could also delete important points.
void StatOutlierRemoval(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud) {
	std::cout << "StatOutlierRemoval Start cloud: " << cloud->size() << std::endl;
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(6);
	//If the theshold is set too low then acceptable points may be removed.
	sor.setStddevMulThresh(3.0);
	sor.filter(*cloud);
	cout << "StatOutlierRemoval End cloud: " << cloud->size() << endl;
}

//Finds edges in point cloud for removal. Necessary when analyzing complex #D input data
//based on https://github.com/denabazazian/Edge_Extraction
// Fast and Robust Edge Extraction in Unorganized Point Clouds (Dena Bazazian, Josep R Casas, Javier Ruiz-Hidalgo) - DICTA2015 
void EdgeDetection(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, std::string dir) {
	std::cout << "EdgeDetection Start cloud:" << cloud->size() << std::endl;

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Normals(new pcl::PointCloud<pcl::PointXYZRGBA>);
	Normals->resize(cloud->size());

	// K nearest neighbor search
	int KNumbersNeighbor = 120; // numbers of neighbors  120                              <--------- Edit here
	float radius = 0.0045; //0.004                                                       <--------- Edit here
	std::vector<int> NeighborsKNSearch;// (KNumbersNeighbor);
	std::vector<float> NeighborsKNSquaredDistance;// (KNumbersNeighbor);
	int* NumbersNeighbor = new  int[cloud->points.size()];
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZRGBA searchPoint;
	double* DLS = new  double[cloud->points.size()];
	double* DLM = new  double[cloud->points.size()];
	double* DMS = new  double[cloud->points.size()];
	double* Sigma = new  double[cloud->points.size()];
	//  ************ All the Points of the cloud *******************
	for (size_t i = 0; i < cloud->points.size(); ++i) {

		//Parallel changes
		//std::vector<int> NeighborsKNSearch;// (KNumbersNeighbor);
		//std::vector<float> NeighborsKNSquaredDistance;// (KNumbersNeighbor);
		//int* NumbersNeighbor = new  int[cloud->points.size()];
		//pcl::PointXYZRGBA searchPoint;

		//END


		searchPoint.x = cloud->points[i].x;
		searchPoint.y = cloud->points[i].y;
		searchPoint.z = cloud->points[i].z;

		if (radius > 0 && /*kdtree.nearestKSearch(searchPoint, KNumbersNeighbor, NeighborsKNSearch, NeighborsKNSquaredDistance)*/ kdtree.radiusSearch(searchPoint, radius, NeighborsKNSearch, NeighborsKNSquaredDistance) > 7) {
			NumbersNeighbor[i] = NeighborsKNSearch.size();
			//cout << "radius" << endl;
		}
		else if (kdtree.nearestKSearch(searchPoint, KNumbersNeighbor, NeighborsKNSearch, NeighborsKNSquaredDistance) > 0) {
			NumbersNeighbor[i] = NeighborsKNSearch.size();
			//cout << "knn" << endl;
		}
		else { NumbersNeighbor[i] = 0; }

		float Xmean; float Ymean; float Zmean;
		float sum = 0.00;
		// Computing Covariance Matrix
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += cloud->points[NeighborsKNSearch[ii]].x;
		}
		Xmean = sum / NumbersNeighbor[i];
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += cloud->points[NeighborsKNSearch[ii]].y;
		}
		Ymean = sum / NumbersNeighbor[i];
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += cloud->points[NeighborsKNSearch[ii]].z;
		}
		Zmean = sum / NumbersNeighbor[i];

		float	CovXX;  float CovXY; float CovXZ; float CovYX; float CovYY; float CovYZ; float CovZX; float CovZY; float CovZZ;

		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].x - Xmean));
		}
		CovXX = sum / (NumbersNeighbor[i] - 1);

		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].y - Ymean));
		}
		CovXY = sum / (NumbersNeighbor[i] - 1);

		CovYX = CovXY;

		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
		}
		CovXZ = sum / (NumbersNeighbor[i] - 1);

		CovZX = CovXZ;

		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += ((cloud->points[NeighborsKNSearch[ii]].y - Ymean) * (cloud->points[NeighborsKNSearch[ii]].y - Ymean));
		}
		CovYY = sum / (NumbersNeighbor[i] - 1);

		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += ((cloud->points[NeighborsKNSearch[ii]].y - Ymean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
		}
		CovYZ = sum / (NumbersNeighbor[i] - 1);

		CovZY = CovYZ;

		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += ((cloud->points[NeighborsKNSearch[ii]].z - Zmean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
		}
		CovZZ = sum / (NumbersNeighbor[i] - 1);

		// Computing Eigenvalue and EigenVector
		Matrix3f Cov;
		Cov << CovXX, CovXY, CovXZ, CovYX, CovYY, CovYZ, CovZX, CovZY, CovZZ;

		SelfAdjointEigenSolver<Matrix3f> eigensolver(Cov);
		if (eigensolver.info() != Success) abort();

		double EigenValue1 = eigensolver.eigenvalues()[0];
		double EigenValue2 = eigensolver.eigenvalues()[1];
		double EigenValue3 = eigensolver.eigenvalues()[2];

		double Smallest = 0.00; double Middle = 0.00; double Largest = 0.00;
		if (EigenValue1 < EigenValue2) { Smallest = EigenValue1; }
		else { Smallest = EigenValue2; }
		if (EigenValue3 < Smallest) { Smallest = EigenValue3; }


		if (EigenValue1 <= EigenValue2 && EigenValue1 <= EigenValue3) {
			Smallest = EigenValue1;
			if (EigenValue2 <= EigenValue3) { Middle = EigenValue2; Largest = EigenValue3; }
			else { Middle = EigenValue3; Largest = EigenValue2; }
		}

		if (EigenValue1 >= EigenValue2 && EigenValue1 >= EigenValue3)
		{
			Largest = EigenValue1;
			if (EigenValue2 <= EigenValue3) { Smallest = EigenValue2; Middle = EigenValue3; }
			else { Smallest = EigenValue3; Middle = EigenValue2; }
		}

		if ((EigenValue1 >= EigenValue2 && EigenValue1 <= EigenValue3) || (EigenValue1 <= EigenValue2 && EigenValue1 >= EigenValue3))
		{
			Middle = EigenValue1;
			if (EigenValue2 >= EigenValue3) { Largest = EigenValue2; Smallest = EigenValue3; }
			else { Largest = EigenValue3; Smallest = EigenValue2; }
		}

		//SmallestEigen[i] = Smallest;
		//MiddleEigen[i] = Middle;
		//LargestEigen[i] = Largest;

		//DLS[i] = std::abs(SmallestEigen[i] / LargestEigen[i]);          // std::abs ( LargestEigen[i] -  SmallestEigen[i] ) ;
		//DLM[i] = std::abs(MiddleEigen[i] / LargestEigen[i]);             // std::abs (  LargestEigen[i] - MiddleEigen[i] ) ;
		//DMS[i] = std::abs(SmallestEigen[i] / MiddleEigen[i]);       // std::abs ( MiddleEigen[i] -  SmallestEigen[i] ) ;
		//Sigma[i] = (SmallestEigen[i]) / (SmallestEigen[i] + MiddleEigen[i] + LargestEigen[i]);

		DLS[i] = std::abs(Smallest / Largest);          // std::abs ( LargestEigen[i] -  SmallestEigen[i] ) ;
		DLM[i] = std::abs(Middle / Largest);             // std::abs (  LargestEigen[i] - MiddleEigen[i] ) ;
		DMS[i] = std::abs(Smallest / Middle);       // std::abs ( MiddleEigen[i] -  SmallestEigen[i] ) ;
		Sigma[i] = (Smallest) / (Smallest + Middle + Largest);
	} // For each point of the cloud
		//});
		//});
	// Color Map For the difference of the eigen values
	double MaxD = 0.00;
	double MinD = cloud->points.size();
	int Ncolors = 256;
	for (size_t i = 0; i < cloud->points.size(); ++i) {
		if (Sigma[i] < MinD) MinD = Sigma[i];
		if (Sigma[i] > MaxD) MaxD = Sigma[i];
	}
	int Edgepoints = 0;
	// Red and white

	for (size_t i = 0; i < cloud->points.size(); ++i) {
		cloud->points[i].r = 240;
		cloud->points[i].g = 230;
		cloud->points[i].b = 140;
		cloud->points[i].a = 255;
	}
	int level = 0;
	float step = ((MaxD - MinD) / Ncolors);
	for (size_t i = 0; i < cloud->points.size(); ++i) {
		if (Sigma[i] > (MinD + (6 * step))) {  //9*step                           <--------- Edit here
			cloud->points[i].r = 192;
			cloud->points[i].g = 192;
			cloud->points[i].b = 192;
			cloud->points[i].a = 0;
			Edgepoints++;
		}
	}
	pcl::PCDWriter writePCD;
	//if (saveClouds) {
	//	writePCD.write(dir + "EdgesMarked.pcd", *cloud, false);
	//}
	std::cout << "EdgeDetection End  Number of Edge points  is :" << Edgepoints << std::endl;
}

//Load point cloud from asci text file
//https://stackoverflow.com/questions/53067067/reading-ascii-point-clouds-in-x-y-z-r-g-b-format
bool loadAsciCloud(std::string filename, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
	std::cout << "loadAsciCloud Start" << std::endl;
	std::cout << filename.c_str() << std::endl;
	FILE* f = fopen(filename.c_str(), "r");
	if (NULL == f)
	{
		std::cout << "ERROR: failed to open file: " << filename << endl;
		return false;
	}
	float x, y, z;
	float r, g, b;
	while (!feof(f))
	{
		int n_args = fscanf(f, "%f %f %f %f %f %f", &x, &y, &z, &r, &g, &b);
		if (n_args != 6)
			continue;
		pcl::PointXYZRGBA point;
		point.x = x;
		point.y = y;
		point.z = z;
		cloud->push_back(point);
	}
	fclose(f);
	std::cout << "loadAsciCloud End  Loaded cloud with " << cloud->size() << " points." << std::endl;
	return true;
}
