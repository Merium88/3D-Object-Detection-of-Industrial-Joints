 #include <ros/ros.h>
 // PCL specific includes

#include <iostream>
#include <string>
#include <vector>
#include <pcl/console/parse.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc

 #include <sensor_msgs/PointCloud2.h>
 #include <pcl_conversions/pcl_conversions.h>
 #include <pcl/point_cloud.h>
 #include <pcl/filters/voxel_grid.h>
 #include <pcl_ros/point_cloud.h>
 #include <pcl/visualization/cloud_viewer.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/features/don.h>
#include <pcl/filters/passthrough.h>

 using namespace pcl;
 using namespace std;

 typedef pcl::PointXYZ PointT;
 typedef pcl::PointCloud<PointT> PointCloudT;

int segment (const PointCloudT::Ptr org_cloud);

int load_ref (int argc, char** argv)
  {
 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    //  std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, "pcd");

    //std::string filename = argv[pcd_filename_indices[0]];
/*
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (filename, *cloud) == -1) //* 
  {
    PCL_ERROR ("Couldn't read file refvrep_pcd.pcd \n");
    return (-1);
  }*/
 if (pcl::io::loadPCDFile ("/home/nus/catkin_ws/Workshop_scene/scene1.pcd", *cloud) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    //showHelp (argv[0]);
    //return (-1);
  }
  //Visualization
  pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
  viewer.showCloud (cloud);
  //call segmentation
    ///////////////////segment(cloud);
   while (!viewer.wasStopped ())
   {
    }
  return (0);
  }

int segment (const PointCloudT::Ptr org_cloud)
{
  pcl::PCDWriter writer;
 pcl::PointCloud<pcl::PointXYZ> cloud_filtered;
  pcl::PointCloud<pcl::PointXYZ> cloud_filtered_1;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>(cloud_filtered));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final (new pcl::PointCloud<pcl::PointXYZ>(cloud_filtered_1));
  cloud_filtered.width = 640;
  cloud_filtered.height = 480;
  cloud_filtered.is_dense = false;
  cloud_filtered.points.resize (cloud_filtered.width * cloud_filtered.height);
  //std::cout<<"points "<<org_cloud->points[200,400]<<std::endl;
  int n,m,w;n = 1;m=0;
  int min_x,min_y,x,y;
  std::cout<<"reached here"<<std::endl;
/*  min_x=10;min_y=10;x=20;y=40; 
for(int n = 1;n<480;n++)
{
 for(int i = 0+m;i<=640*n;i++)
  {
     
       if(i>=min_x+m && i<=(min_x+x)+m && n>=min_y && n<=(min_y+y))
       {
        cloud_filtered.points[i].x = org_cloud->points[i].x;
        cloud_filtered.points[i].y = org_cloud->points[i].y;
        cloud_filtered.points[i].z = org_cloud->points[i].z;
         // std::cout<<"points x "<<org_cloud->points[i].x<<"y "<<org_cloud->points[i].y<<"z "<<org_cloud->points[i].z<<std::endl;
       } 
       else  
     {
     cloud_filtered.points[i].x = 0;//org_cloud->points[i].x;
     cloud_filtered.points[i].y = 0;//org_cloud->points[i].y;
     cloud_filtered.points[i].z = 0;//org_cloud->points[i].z;
     }
  }
m=m+640;
}
  */
  
 // Create the filtering object for chord
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (org_cloud);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (-2, 2);
  //pass.setFilterLimitsNegative (true);
  pass.filter (cloud_filtered);
   *cloud = cloud_filtered;
   
 /* pcl::PassThrough<pcl::PointXYZ> pass1;
  pass1.setInputCloud (cloud);
  pass1.setFilterFieldName ("y");
  pass1.setFilterLimits (-0.25, 0.2);
  */
  pcl::PassThrough<pcl::PointXYZ> pass1;
  pass1.setInputCloud (cloud);
  pass1.setFilterFieldName ("z");
  pass1.setFilterLimits (-2, 2);
  //pass.setFilterLimitsNegative (true);
  pass1.filter (cloud_filtered_1);
  // Create the filtering object for chord
  /*  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (org_cloud);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (-0.35, -0.05);
  //pass.setFilterLimitsNegative (true);
  pass.filter (cloud_filtered);
   *cloud = cloud_filtered;
   
  pcl::PassThrough<pcl::PointXYZ> pass1;
  pass1.setInputCloud (cloud);
  pass1.setFilterFieldName ("y");
  pass1.setFilterLimits (-0.25, 0.25);
  //pass.setFilterLimitsNegative (true);
  pass1.filter (cloud_filtered_1);*/
  std::cout<<"what is size"<<cloud_filtered.points.size ()<<std::endl;
   *cloud_final = cloud_filtered_1;
  // Visualization
     pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
     viewer.showCloud (cloud_final);

    while (!viewer.wasStopped ())
     {
     }
  //writer.write ("only_stub_5.pcd", cloud_filtered_1, false);  
 return 0;
}
void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      int num;
     
      // Container for original & filtered data
      pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2; 
      pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
      pcl::PCLPointCloud2 cloud_filtered;
      pcl::PointCloud<pcl::PointXYZRGB> cloudvrep;

 
     // Convert to PCL data type
     pcl_conversions::toPCL(*cloud_msg, *cloud);
     pcl::fromROSMsg(*cloud_msg, cloudvrep);
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>(cloudvrep));
     
     // Save reference cloud
     pcl::io::savePCDFileASCII ("/home/nus/catkin_ws/Workshop_scene/scene20.pcd", cloudvrep);  

     //Visualization
     pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
     viewer.showCloud (point_cloud_ptr);

     while (!viewer.wasStopped ())
     {
     }
    
     //std::cout<<cloudvrep.width<<std::endl;
    
   }

  int main (int argc, char** argv)
   {
     // Initialize ROS
     ros::init (argc, argv, "my_pcl_tutorial");
     ros::NodeHandle nh;

     // Create a ROS subscriber for the input point cloud "depth"
     //ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points", 1, cloud_cb);
   
     load_ref(argc,argv);
     // Spin
     ros::spin ();
   }

