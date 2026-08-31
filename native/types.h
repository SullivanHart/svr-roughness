#ifndef TYPES_H
#define TYPES_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/IO/OFF.h>

#include <CGAL/Point_set_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/Poisson_reconstruction_function.h>
#include <CGAL/property_map.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/IO/OFF.h>
#include <CGAL/AABB_tree.h>
#if __has_include(<CGAL/AABB_traits_3.h>)
#include <CGAL/AABB_traits_3.h>
#else
#include <CGAL/AABB_traits.h>
#endif
#include <CGAL/AABB_halfedge_graph_segment_primitive.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Surface_mesh_shortest_path/Surface_mesh_shortest_path.h>
#include <CGAL/Surface_mesh_shortest_path.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Surface_mesh.h>

// Types
typedef CGAL::Simple_cartesian<double> Kernel2;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point;
//typedef Kernel2::Point_3 Point2;
typedef CGAL::Exact_predicates_inexact_constructions_kernel::Vector_3 CGAL_Vector;
typedef CGAL::Point_set_3<Point, CGAL_Vector> Point_set;
typedef std::pair<Point, CGAL_Vector> Point_with_normal;
typedef CGAL::First_of_pair_property_map<Point_with_normal> Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;

typedef Kernel::Sphere_3 Sphere;
typedef std::vector<Point_with_normal> PointList;

//typedef CGAL::Polyhedron_3<Kernel2> Polyhedron;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> Polyhedron;
typedef CGAL::Poisson_reconstruction_function<Kernel> Poisson_reconstruction_function;
typedef CGAL::Surface_mesh_default_triangulation_3 STr;
typedef CGAL::Surface_mesh_complex_2_in_triangulation_3<STr> C2t3;
typedef CGAL::Implicit_surface_3<Kernel, Poisson_reconstruction_function> Surface_3;

// AABB_tree typedefs
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron, CGAL::Default, CGAL::Tag_false> Primitive;
typedef CGAL::AABB_traits_3<Kernel, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> AABBPrimitive;
typedef CGAL::AABB_traits_3<Kernel, AABBPrimitive> AABBTraits;
typedef CGAL::AABB_tree<AABBTraits> AABB_tree;

typedef CGAL::Surface_mesh_shortest_path_traits<Kernel, Polyhedron> Traits2;
typedef CGAL::Surface_mesh_shortest_path<Traits2> Surface_mesh_shortest_path;
typedef CGAL::Surface_mesh_shortest_path<Traits2>::Face_location Face_location;
typedef CGAL::Surface_mesh_shortest_path<Traits2>::Shortest_path_result Shortest_path_result;
typedef CGAL::Surface_mesh<Kernel::Point_3> Mesh2;
typedef boost::graph_traits<Polyhedron> Graph_traits;
typedef boost::graph_traits<Mesh2>::halfedge_descriptor halfedge_descriptor;
typedef boost::graph_traits<Mesh2>::edge_descriptor     edge_descriptor;
typedef boost::graph_traits<Polyhedron>::halfedge_descriptor halfedge_descriptor2;
typedef boost::graph_traits<Polyhedron>::edge_descriptor     edge_descriptor2;
typedef Graph_traits::face_iterator face_iterator;
typedef Tree::Point_and_primitive_id Point_and_primitive_id;

typedef CGAL::Search_traits_3<Kernel> KdTraits;
typedef CGAL::Kd_tree<KdTraits> KdTree2;
typedef CGAL::Fuzzy_sphere<KdTraits> Fuzzy_sphere;
typedef CGAL::Sliding_midpoint<KdTraits>              Sliding_midpoint;

typedef std::vector< std::vector<double> > DoubleMatrix;
typedef std::vector<double> Row;
typedef std::vector< std::vector<int> > MatrixInt;
typedef std::vector<int> RowInt;

//Nontraditional Roughness Calculation
typedef Kernel::Segment_3 Segment;
typedef boost::optional< Tree::Intersection_and_primitive_id<Segment>::Type > Segment_intersection;

// Concurrency
//#ifdef CGAL_LINKED_WITH_TBB
typedef CGAL::Sequential_tag Concurrency_tag;

//#else
//typedef CGAL::Sequential_tag Concurrency_tag;
//#endif

#endif
