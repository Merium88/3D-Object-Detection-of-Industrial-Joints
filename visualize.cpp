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
pcl::PointCloud<pcl::PointXYZRGBA> cloudV;
//Visualization
pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
//viewer.setBackgroundColor (255, 255, 255);
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
        
     pcl::copyPointCloud(cloudrgb,cloudV);   
     std::cout<< "REached here"<<std::endl;
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudVPtr(new pcl::PointCloud<pcl::PointXYZRGBA>(cloudV));
     
    pcl::visualization::PointCloudColorHandlerCustom<PointType> color_handler (cloudVPtr, 200, 0, 0);
    viewer.updatePointCloud (cloudVPtr, color_handler, "Edges keypoints");
}
	
	
int main(int argc, char** argv)
  {
    // Initialize ROS
    //ros::init (argc, argv, "visualize");
    //ros::NodeHandle nh;
   for (int l = 0; l < 42; ++l)
   {
   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model (new pcl::PointCloud<pcl::PointXYZRGBA>());
   string filename;
   cout<<"pose no: "<<l<<endl;
   string st1 = "/home/nus/catkin_ws/iros_data/";
   stringstream sm;sm << l;
   string st2 = ".pcd";   filename = st1 + sm.str() + st2;
   if (pcl::io::loadPCDFile (filename, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    //showHelp (argv[0]);
    //return (-1);
  }
    //Visualization
    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
    viewer.setBackgroundColor (255, 255, 255);
    
    //while (!viewer.wasStopped ())
    //{
     // Create a ROS subscriber for the input point cloud "depth"
    //sub = nh.subscribe ("Edge_PCD", 1, edge_detect);
    //pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudVPtr(new pcl::PointCloud<pcl::PointXYZRGBA>(cloudV));
    pcl::visualization::PointCloudColorHandlerCustom<PointType> color_handler (model, 0, 0, 0);
    viewer.addPointCloud  (model, color_handler, "edges");
     while (!viewer.wasStopped ())
    {
    viewer.spin();
    }
    //viewer.spinOnce();
    // Spin
     //ros::spin ();
     
    }
    
     
     
    return 1;
  }
