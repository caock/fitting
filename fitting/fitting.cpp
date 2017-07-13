#include <iostream>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h> // for Randome add point
#include <CGAL/Fuzzy_sphere.h>		 // for radius neighbor search
#include <CGAL/Kd_tree.h>
#include <CGAL/bounding_box.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>	// for inverse
#include <limits>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Search_traits_3<Kernel> Traits;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;
typedef CGAL::Kd_tree<Traits>      Tree;

std::vector<Point> RandomSphere()
{
	const std::size_t PointNum = 1000;
	// generator for random data points in the square ( (-1,-1), (1,1) )
	CGAL::Random rnd((unsigned int)time(0));
	CGAL::Random_points_in_cube_3<Kernel::Point_3> rpit(1.0, rnd);
	Tree tree;
	for (int i=0;i < PointNum;i++)
	{
		tree.insert(Point(*rpit++));
	}
	Point center(0, 0, 0);
	double radius = 0.6;
	Fuzzy_sphere fs(center, radius, 0.1);
	std::vector<Point> sphere;
	tree.search(std::back_inserter(sphere), fs);
	
	return sphere;
}

void CheckJacobi(std::function<Eigen::VectorXd(Eigen::VectorXd)> getF, std::function<Eigen::MatrixXd(Eigen::VectorXd)> getJ)
{
	const double eps = sqrt(std::numeric_limits<double>::epsilon());
	Eigen::Vector4d value = Eigen::Vector4d::Random();
	const std::size_t N = value.size();
	for (std::size_t i = 0; i < N; i++)
	{
		Eigen::Vector4d dvalue = value;
		dvalue[i] += eps;
		Eigen::VectorXd dF = (getF(dvalue) - getF(value)) / eps;
		Eigen::VectorXd Ji = getJ(value).col(i);
		const std::size_t M = dF.size();
		if (dF.size() != Ji.size() || dF.size() != M)
		{
			std::cout << "dF size = " << dF.size() << "\tJi size = " << Ji.size() << std::endl;
		}
		else
		{
			std::cout << "dF = " << dF.block(0,0,5,1).transpose() << "\nJi = " << Ji.block(0, 0, 5, 1).transpose();
			double d = (dF - Ji).norm();
			std::cout << d << std::endl;
		}
	}
}

void GaussNewtonOneStep(std::function<Eigen::VectorXd(Eigen::VectorXd)> getF, std::function<Eigen::MatrixXd(Eigen::VectorXd)> getJ, Eigen::VectorXd& value)
{
	CheckJacobi(getF, getJ);

	Eigen::VectorXd F = getF(value);
	Eigen::MatrixXd J = getJ(value);
	Eigen::MatrixXd Jt = J.transpose();
	Eigen::MatrixXd v = value.transpose();

	value += (Jt*J).inverse() * Jt * F;
}

void fittingSphere(std::vector<Point> sphere)
{
	Eigen::Vector3d center(0.1, 0.2, -0.3);
	Kernel::Iso_cuboid_3 box = CGAL::bounding_box(sphere.begin(), sphere.end());
	double radius = sqrt((box.max() - box.min()).squared_length()) * 0.5;
	//double radius = (box.max().x() - box.min().x()) * 0.5;

	//Eigen::MatrixXd F(sphere.size(), 5, 0.0);
	//for (std::size_t i = 0; i < sphere.size(); i++)
	//{
	//	Point p = sphere[i];
	//	F(i, 0) = (p - Point(0, 0, 0)).squared_length();
	//	F(i, 1) = p.x();
	//	F(i, 2) = p.y();
	//	F(i, 3) = p.z();
	//	F(i, 4) = 1;
	//}
	auto getF = [&sphere](Eigen::VectorXd value)->Eigen::VectorXd
	{
		Eigen::Vector3d z(value[0], value[1], value[2]);
		double radius = value[3];
		Eigen::VectorXd F = Eigen::VectorXd::Zero(sphere.size());
		for (std::size_t i = 0; i < sphere.size(); i++)
		{
			Eigen::Vector3d p = Eigen::Vector3d(sphere[i].x(), sphere[i].y(), sphere[i].z());
			F(i) = (p - z).norm() - radius;
			//F(i) = F(i) * F(i);
		}
		return F;
	};
	auto getJ = [&sphere](Eigen::VectorXd value)->Eigen::MatrixXd
	{
		Eigen::Vector3d z(value[0], value[1], value[2]);
		Eigen::MatrixXd J = Eigen::MatrixXd::Zero(sphere.size(), 4);
		for (std::size_t i = 0; i < sphere.size(); i++)
		{
			Point p = sphere[i];
			double a = z.x() - p.x();
			double b = z.y() - p.y();
			double c = z.z() - p.z();
			double d = sqrt(a*a + b*b + c*c);
			J(i, 0) = (z.x() - p.x()) / d;
			J(i, 1) = (z.y() - p.y()) / d;
			J(i, 2) = (z.z() - p.z()) / d;
			J(i, 3) = -1;
		}
		return J;
	};
	auto getEnergy = [&](Eigen::VectorXd value)->double
	{
		return getF(value).norm();
	};
	Eigen::VectorXd value(4);
	value.x() = center.x();
	value.y() = center.y();
	value.z() = center.z();
	value[3] = radius;
	double beginEnergy = getEnergy(value);
	std::cout << "begin_center = " << center.transpose() << "\t\t\tradius = " << radius << "\tEnergy = " << beginEnergy << std::endl;
	for(int i=0; i < 15; i++)
	{	
		GaussNewtonOneStep(getF, getJ, value);

		center = Eigen::Vector3d(value.x(), value.y(), value.z());
		radius = value[3];
		double curEnergy = getEnergy(value);
		if (curEnergy > beginEnergy)
			std::cout << "stop\t";
		std::cout << "center = " << center.transpose() << "\t\tradius = " << radius << "\tEnergy = " << curEnergy << std::endl;
		beginEnergy = curEnergy;
	}
	
}

int main()
{
	std::vector<Point> sphere = RandomSphere();
	std::cout << "sphere size = " << sphere.size() << std::endl << std::endl;
	fittingSphere(sphere);

	return 0;
}