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
	const std::size_t PointNum = 10000;
	// generator for random data points in the square ( (-1,-1), (1,1) )
	CGAL::Random rnd((unsigned int)time(0));
	CGAL::Random_points_in_cube_3<Kernel::Point_3> rpit(1.0, rnd);
	Tree tree;
	for (int i=0;i < PointNum;i++)
	{
		tree.insert(Point(*rpit++));
	}
	Point center(0, 0, 0);
	double radius = 0.8;
	Fuzzy_sphere fs(center, radius, 0.1);
	std::vector<Point> sphere;
	tree.search(std::back_inserter(sphere), fs);
	
	return sphere;
}

bool CheckJacobi(std::function<double(Eigen::VectorXd)> getF, std::function<Eigen::RowVectorXd(Eigen::VectorXd)> getJ)
{
	const double eps = sqrt(std::numeric_limits<double>::epsilon());
	Eigen::Vector4d value = Eigen::Vector4d::Random();
	const std::size_t Ncase = value.size();
	Eigen::Vector4d dF = Eigen::Vector4d::Zero();
	Eigen::Vector4d J = Eigen::Vector4d::Zero();
	for (std::size_t i = 0; i < Ncase; i++)
	{
		Eigen::Vector4d dvalue = value;
		dvalue[i] += eps;
		dF(i) = (getF(dvalue) - getF(value)) / eps;
		J(i) = getJ(value)(i);
	}
	//std::cout << "dF = " << dF.transpose() << "\nJi = " << J.transpose() << "\nDiff = " << (dF - J).transpose() << std::endl;
	if ((dF - J).norm() < 0.01)
		return true;
	return false;
}

Eigen::VectorXd GaussNewtonOneStep(std::function<double(Eigen::VectorXd)> getF, std::function<Eigen::RowVectorXd(Eigen::VectorXd)> getJ, Eigen::VectorXd value)
{
	if (!CheckJacobi(getF, getJ))
	{
		std::cout << "The Jacobi Not Match!" << std::endl;
		return Eigen::VectorXd::Zero(value.size());
	}

	double F = getF(value);
	Eigen::RowVectorXd J = getJ(value);
	Eigen::VectorXd Jt = J.transpose();
	
	return (Jt*J).inverse() * Jt * F;
	//std::cout << "value = " << value << "\ns = " << (Jt*J).inverse() * Jt * F << std::endl;
	//std::cout << "J = " << J << "\nJt = " << Jt << "\nF = " << F << "\nvalue = " << value << std::endl;
}

void fittingSphere(std::vector<Point> sphere)
{
	Eigen::Vector3d center(0.1, 0.2, -0.13);
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
	auto getF = [&sphere](Eigen::VectorXd value)->double
	{
		Eigen::Vector3d z(value[0], value[1], value[2]);
		double radius = value[3];
		double f = 0.0;
		for (std::size_t i = 0; i < sphere.size(); i++)
		{
			Eigen::Vector3d p = Eigen::Vector3d(sphere[i].x(), sphere[i].y(), sphere[i].z());
			double d = (p - z).norm() - radius;
			f += d * d;
		}
		return f;
	};
	auto getJ = [&sphere](Eigen::VectorXd value)->Eigen::RowVectorXd
	{
		Eigen::Vector3d z(value[0], value[1], value[2]);
		Eigen::VectorXd J = Eigen::VectorXd::Zero(value.size());
		for (std::size_t i = 0; i < sphere.size(); i++)
		{
			Eigen::Vector3d p = Eigen::Vector3d(sphere[i].x(), sphere[i].y(), sphere[i].z());
			double radius = value[3];
			double a = z.x() - p.x();
			double b = z.y() - p.y();
			double c = z.z() - p.z();
			double d = sqrt(a*a + b*b + c*c);
			Eigen::VectorXd Ji = Eigen::VectorXd::Zero(J.size());
			Ji(0) += (z.x() - p.x()) / d;
			Ji(1) += (z.y() - p.y()) / d;
			Ji(2) += (z.z() - p.z()) / d;
			Ji(3) += -1;
			J += 2 * ((p - z).norm() - radius) * Ji;
		}
		return J.transpose();
	};
	auto getEnergy = [&](Eigen::VectorXd value)->double
	{
		return getF(value);
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
		Eigen::VectorXd step = GaussNewtonOneStep(getF, getJ, value);
		std::cout << "step = " << step.transpose() << std::endl;
		value += step;
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