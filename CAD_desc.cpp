#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/filters/impl/uniform_sampling.hpp>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/point_representation.h>
#include <pcl/impl/point_types.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/keypoints/iss_3d.h>


using namespace std;
typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> ColorHandlerT;
std::string model_filename_;
std::string scene_filename_;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (true);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
bool use_SHOT_ (true);
bool plot_descriptors(true);
bool icp_flag (true);
float model_ss_ (0.01f);
float scene_ss_ (0.01f);
float rf_rad_ (0.01f);
float descr_rad_ (0.01f);
float cg_size_ (0.02f);
float cg_thresh_ (2.0f);
ros::Subscriber sub;
void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;
  std::cout << "     -c:                     Show used correspondences." << std::endl;
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  model_filename_ = argv[filenames[0]];
  scene_filename_ = argv[filenames[1]];

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }

  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
}

double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
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

/* virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
*/

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      int num;
      pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
     cout<<"inside"<<endl;
      // Container for original & filtered data
      pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2; 
      pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
      pcl::PCLPointCloud2 cloud_filtered;
      pcl::PointCloud<pcl::PointXYZRGBA> cloudvrep;

 
     // Convert to PCL data type
     pcl_conversions::toPCL(*cloud_msg, *cloud);
     pcl::fromROSMsg(*cloud_msg, cloudvrep);
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGBA>(cloudvrep));

     // Save reference cloud
     //pcl::io::savePCDFileASCII ("asus_scene.pcd", cloudvrep);  

     //std::cout<<cloudvrep.width<<std::endl;
    //return(cloudvrep);
   //////////////////////////////////////////////////////////////////Object Detection Module//////////////////////////////////////////////////// 
     //parseCommandLine (argc, argv);

  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  //pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<PointType>::Ptr full_pose (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr full_pose_transform (new pcl::PointCloud<PointType> ());
  //  Load clouds
  //
string st1 = "/home/nus/catkin_ws/stub_poses/";///home/nus/catkin_ws/stub_poses/3.pcd    front-pose_chord_new.pcd   only_chord.pcd
string st2 = ".pcd";
string st3 = "Partial_View";
string st4 = ".txt";
string filename ;
int h = 0;
  for (int l = 0; l <= 41; ++l)
{
    cout<<"Pose "<< l<< " Matched to Scene "<<endl;
  stringstream ss;
/*if (l==0)
    h = 3;
else if (l==1)
    h = 18;
else if (l==2)
    h = 23;
else if (l==3)
    h = 33;
*/
  ss << l;
  filename = st1 + ss.str() + st2;
  

  if (pcl::io::loadPCDFile (filename, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    //showHelp (argv[0]);
    //return (-1);
  }
 
  //
  //  Set up resolution invariance
  //
  if (use_cloud_resolution_)
  {
    float resolution = static_cast<float> (computeCloudResolution (model));
    if (resolution != 0.0f)
    {
      model_ss_   *= resolution;
      scene_ss_   *= resolution;
      rf_rad_     *= resolution;
      descr_rad_  *= resolution;
      cg_size_    *= resolution;
    }

    std::cout << "Model resolution:       " << resolution << std::endl;
    std::cout << "Model sampling size:    " << model_ss_ << std::endl;
    std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
    std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
    std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
    std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
  }
// REmove NaNs from POintcloud
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*model,*model, indices); 
  pcl::removeNaNFromPointCloud(*scene,*scene, indices); 
  //  Compute Normals
  //
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (40);
  norm_est.setInputCloud (model);
  norm_est.compute (*model_normals);

  norm_est.setInputCloud (scene);
  norm_est.compute (*scene_normals);

  //
  //  Downsample Clouds to Extract keypoints
  //

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.filter (*model_keypoints);
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch (scene_ss_);
  uniform_sampling.filter (*scene_keypoints);
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

 // Add another keypoint extractor
 pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
/*
 // Compute model_resolution
	pcl::ISSKeypoint3D<pcl::PointXYZRGBA, pcl::PointXYZRGBA> iss_detector;
	iss_detector.setSearchMethod (tree);
    float model_resolution = static_cast<float> (computeCloudResolution (model));
	iss_detector.setSalientRadius (6 * model_resolution);
	iss_detector.setNonMaxRadius (4 * model_resolution);
	iss_detector.setThreshold21 (1.75);//0.975
	iss_detector.setThreshold32 (1.75);//0.975
	iss_detector.setMinNeighbors (5);
	iss_detector.setNumberOfThreads (4);
	iss_detector.setInputCloud (model);
	iss_detector.compute (*model_keypoints);
	std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
	 
	 float scene_resolution = static_cast<float> (computeCloudResolution (scene));
	iss_detector.setSearchMethod (tree);
	iss_detector.setSalientRadius (6 * scene_resolution);
	iss_detector.setNonMaxRadius (4 * scene_resolution);
	iss_detector.setThreshold21 (1.75);
	iss_detector.setThreshold32 (1.75);
	iss_detector.setMinNeighbors (5);
	iss_detector.setNumberOfThreads (4);
	iss_detector.setInputCloud (scene);
	iss_detector.compute (*scene_keypoints);
	  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
	  
 */
 
  //  Compute Descriptor for keypoints
  //
  if (use_SHOT_)
{
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);

  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);

  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene);
  descr_est.compute (*scene_descriptors);
  
  //Write descriptors to a file
  float shot_desc ;
  ofstream myfile;
  string filename2;
  filename2 = st3 + ss.str() + st4;
  myfile.open (filename2.c_str());


  for (size_t i = 0; i < model_descriptors->size (); ++i)
  {
  for (size_t j = 0; j < 352; ++j)
  {
 //copyToFloatArray(model_descriptors,*shot_desc);
  //std::cout << "SHOT Descriptors for point: " << i<< "descriptor no: "<<j<<" value is "<<model_descriptors->at (i).descriptor[j]<< std::endl;
  myfile <<model_descriptors->at (i).descriptor[j]<< std::endl;
  }
  }
 /* myfile.close();
  myfile.open ("SHOTScene.txt");
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
  for (size_t j = 0; j < 352; ++j)
  {
 //copyToFloatArray(model_descriptors,*shot_desc);
  //std::cout << "SHOT Descriptors for point: " << i<< "descriptor no: "<<j<<" value is "<<model_descriptors->at (i).descriptor[j]<< std::endl;
  myfile <<scene_descriptors->at (i).descriptor[j]<< std::endl;
  }
  }
  myfile.close();*/
}
}

   std::cout<<"exiting object recognition loop "<<std::endl;
   sub.shutdown();
   }

int main (int argc, char** argv)
   {
     // Initialize ROS
     ros::init (argc, argv, "my_pcl_tutorial");
     ros::NodeHandle nh;  
     // Create a ROS subscriber for the input point cloud "depth"
     sub = nh.subscribe ("/camera/depth_registered/points", 1, cloud_cb);
       // Spin
     ros::spin ();
   
   }
