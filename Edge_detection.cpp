#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h> 
#include <time.h>
#include <stdio.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/filters/impl/uniform_sampling.hpp>

using namespace std;
typedef pcl::PointXYZRGBA PointType;
ros::Subscriber sub;
ros::Publisher pub;

time_t start,end;
float radius=0.002;
float model_ss_ (0.01f);

double computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

 pcl::PointCloud<PointType>::Ptr downsample(const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
    //Downsample pointcloud
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr result (new pcl::PointCloud<PointType> ());
     pcl::UniformSampling<PointType> uniform_sampling;
     uniform_sampling.setInputCloud (cloud);
     uniform_sampling.setRadiusSearch (radius);
     uniform_sampling.filter (*result);
     return result;

}

void edge_detect (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
      // Container for original & filtered data
      pcl::PCLPointCloud2* cloud_sent = new pcl::PCLPointCloud2; 
      pcl::PCLPointCloud2ConstPtr cloudPtr(cloud_sent);
      pcl::PointCloud<pcl::PointXYZRGBA> cloudrgb;
      sensor_msgs::PointCloud2 output;
 
     // Convert to PCL data type
     pcl_conversions::toPCL(*cloud_msg, *cloud_sent);
     pcl::fromROSMsg(*cloud_msg, cloudrgb);
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>(cloudrgb));
     
     // Read pointcloud
     pcl::PointCloud<PointType>::Ptr Edges (new pcl::PointCloud<PointType>());
    // pcl::PCLPointCloud2 Edge_points = new pcl::PCLPointCloud2;
    // pcl::PCLPointCloud2ConstPtr Edgeptr(Edge_points);
     pcl::PointCloud<PointType>::Ptr off_scene_cloud_keypoints (new pcl::PointCloud<PointType> ());
         
     std::cout<< "REached here"<<std::endl;

     //Compute cloud resolution
     float resolution = static_cast<float> (computeCloudResolution (cloud));
     if (resolution != 0.0f)
     {
      model_ss_   *= resolution;
     }
     std::cout << "Model resolution:       " << resolution << std::endl;

     //Downsample cloud
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_keypoints = downsample(cloud);
     std::cout<< "Size of original pointcloud "<<cloud->size ()<<std::endl;
     std::cout<< "Size of downsampled pointcloud "<<cloud_keypoints->size ()<<std::endl;

     //Search for NN with in a range
     time (&start);
     pcl::KdTreeFLANN<PointType> kdtree;
     kdtree.setInputCloud (cloud_keypoints);
     float range = 2.0f * rand () / (RAND_MAX + 1.0f);
     std::vector<float> shift_x,shift_y,shift_z;
     std::cout<<"what is range for search for knn "<<range<<std::endl;
     int counter = 0;      
     for (size_t i = 0; i < cloud_keypoints->size (); ++i)
     {
     std::vector<int> neigh_indices;
     std::vector<float> neigh_sqr_dists;
     int found_neighs = kdtree.nearestKSearch (cloud_keypoints->at (i), 100, neigh_indices, neigh_sqr_dists); //search with in k nearest neighbors
     Eigen::Vector4f centroid; 
     pcl::PointCloud<PointType>::Ptr NN (new pcl::PointCloud<PointType>());
     NN->points.resize(neigh_indices.size ());

        for (size_t j = 0; j < neigh_indices.size (); j++)
        {
        //std::cout << "Size of neighbors " << neigh_indices.size ()<<"  "<<cloud_keypoints->at (neigh_indices[j]).x<<std::endl;
        NN->points[j].x = cloud_keypoints->at (neigh_indices[j]).x;        
        NN->points[j].y = cloud_keypoints->at (neigh_indices[j]).y;
        NN->points[j].z = cloud_keypoints->at (neigh_indices[j]).z;

        } 

      pcl::compute3DCentroid(*NN,centroid);
    
    float mag = abs(centroid[0]-cloud_keypoints->at (i).x)+abs(centroid[1]-cloud_keypoints->at (i).y)+abs(centroid[2]-cloud_keypoints->at (i).z);
    Edges->points.resize(counter+1);
    //std::cout<<"Size of mag "<<mag<<std::endl;
    if(mag>0.004)
	  {
    Edges->at(counter).x = cloud_keypoints->at (i).x;
    Edges->at(counter).y = cloud_keypoints->at (i).y;
    Edges->at(counter).z = cloud_keypoints->at (i).z;
    counter = counter + 1; 
	  }
     }
    time (&end);
    double dif = difftime (end,start);
    std::cout<<"Elasped time is "<< dif<<std::endl;
     std::cout<< "Size of Edges "<<Edges->size ()<<std::endl;
 /*   //Visualization
    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
    viewer.setBackgroundColor (255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<PointType> color_handler (Edges, 0, 0, 0);
    viewer.addPointCloud (Edges, color_handler, "Edges keypoints");
    pcl::transformPointCloud (*cloud_keypoints, *off_scene_cloud_keypoints, Eigen::Vector3f (0,2,2), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::visualization::PointCloudColorHandlerCustom<PointType> color_handler_kp (off_scene_cloud_keypoints, 0, 0, 200);
    viewer.addPointCloud (off_scene_cloud_keypoints, color_handler_kp, "Full keypoints");
    while (!viewer.wasStopped ())
    {
    viewer.spin();
    }*/
    
        //output = Edges;
        pub.publish (*Edges);


}
	
	
int main(int argc, char** argv)
  {
     // Initialize ROS
     ros::init (argc, argv, "my_pcl_tutorial");
     ros::NodeHandle nh;
     // Create a ROS subscriber for the input point cloud "depth"
     sub = nh.subscribe ("/camera/depth_registered/points", 1, edge_detect);
     //PUblish edges pointcloud
     pub = nh.advertise<sensor_msgs::PointCloud2> ("Edge_PCD", 1);
     // Spin
     ros::spin ();
     
     
    return 1;
  }
