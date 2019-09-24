#include <iostream>
#include <string>
#include <sstream>
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
#include <cmath>
#include <numeric>
#include <vector>
#include <functional>
#include <valarray>


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
float cg_size_ (0.03f);
float cg_thresh_ (3.0f);
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

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr icp_align(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr target, pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr source)
{
  pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
  icp.setMaximumIterations (100);
  icp.setInputSource (source);
  icp.setInputTarget (target);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_icp (new pcl::PointCloud<pcl::PointXYZRGBA>); 
  icp.align (*cloud_icp);
  double score = icp.getFitnessScore();
  printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
  if (score <= 0.001)
  {
    icp_flag = false;
  }
  return(cloud_icp);
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
  //  Load clouds
  //
string st1 = "/home/nus/catkin_ws/stub_poses/";
string st2 = ".pcd";
string filename ;
int h = 0;
  for (int l = 0; l < 1; ++l)
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
 /* if (pcl::io::loadPCDFile ("/home/nus/catkin_ws/asus_scene.pcd", *scene) < 0)
  {
    std::cout << "Error loading scene cloud." << std::endl;
    //showHelp (argv[0]);
   // return (-1);
  }*/
  //scene = point_cloud_ptr ;

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
 
  //
  //  Downsample Clouds to Extract keypoints
  //
  std::vector<int>  keyp_indices;
  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.filter (*model_keypoints);
  //indices = uniform_sampling.getKeypointsIndices ();

  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch (scene_ss_);
  uniform_sampling.filter (*scene_keypoints);
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
 //  Compute Normals
  //
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (40);
  norm_est.setInputCloud (model_keypoints);
  norm_est.compute (*model_normals);

  norm_est.setInputCloud (scene_keypoints);
  norm_est.compute (*scene_normals);

 // Calculate Average of Surface Normals for each keypoint
   pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
   kdtree.setInputCloud (model_keypoints);
//////////////////////////////////////////////////////////////Calculate Variance Based Descriptor for Model Cloud//////////////////////////
  // Stores nearest neighbors in the cloud 
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
   float radius = 0.05;//256.0f * rand () / (RAND_MAX + 1.0f);

  float NNNaverage = 0;
  float NNNvariance = 0;
  float var_dec_model[model_keypoints->size ()][3];
for (int t = 0;t < (model_keypoints->size ());t++)//
{ 
 float avg_dec[3];
       
 std::cout<<"t  "<<t<<std::endl;
 
 for(int u = 0;u<3;u++) // how many times radius for search is varied
 {
  //keyp [t]= model_keypoints->at (t);
  //std::cout<<"Keypoint size "<< model_keypoints->size ()<<"normal size "<<model_normals->size ()<<std::endl;

   // Neighbors within radius search
if ( kdtree.radiusSearch (model_keypoints->at (t), radius*(u+1), pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
  {
  //Create parameters for cos-theta calculation 
  float Dot_prod[pointIdxRadiusSearch.size ()];
  float norm_vect[pointIdxRadiusSearch.size ()];
  float norm_keyp[pointIdxRadiusSearch.size ()];
  float theta[pointIdxRadiusSearch.size ()];
  
  std::cout<<"searched index "<< pointIdxRadiusSearch.size ()<<std::endl;
  
    for (size_t i = 0; i <pointIdxRadiusSearch.size (); ++i)//
    {    
     /* std::cout << "    "  <<   model_keypoints->points[ pointIdxRadiusSearch[i] ].x 
                << " " << model_keypoints->points[ pointIdxRadiusSearch[i] ].y 
                << " " << model_keypoints->points[ pointIdxRadiusSearch[i] ].z 
                << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
     */
     //Create vector to store normals of nearest neighbors
     //std::cout<<"normal value is "<<model_normals->at(pointIdxRadiusSearch[i]).normal_x<<std::endl;
    float norm_vect_x = model_normals->at(pointIdxRadiusSearch[i]).normal_x;
    float norm_vect_y = model_normals->at(pointIdxRadiusSearch[i]).normal_y;
    float norm_vect_z = model_normals->at(pointIdxRadiusSearch[i]).normal_z;
    float keyp_vect_x = model_normals->at(t).normal_x;
    float keyp_vect_y = model_normals->at(t).normal_y;
    float keyp_vect_z = model_normals->at(t).normal_z;
    //Components to calculate cos theta................Code not optimized
    Dot_prod[i]=(norm_vect_x*keyp_vect_x+norm_vect_y*keyp_vect_y+norm_vect_z*keyp_vect_z);
    norm_vect[i] = pow((norm_vect_x*norm_vect_x+norm_vect_y*norm_vect_y+norm_vect_z*norm_vect_z),0.5);
    norm_keyp[i] = pow((keyp_vect_x*keyp_vect_x+keyp_vect_y*keyp_vect_y+keyp_vect_z*keyp_vect_z),0.5);
    theta[i] = Dot_prod[i]/(norm_vect[i]*norm_keyp[i]);
    //std::cout<<" "<<theta[i]<<std::endl;
    
    //Calculate sum of all thetas to get average
     NNNaverage = NNNaverage + theta[i]; 
     }
     NNNaverage = NNNaverage/pointIdxRadiusSearch.size ();

     //Calculate variance 
      for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
    {
      NNNvariance = NNNvariance + pow((NNNaverage-theta[i]),2);
     }
     NNNvariance = NNNvariance/pointIdxRadiusSearch.size ();
    // std::cout<<"Average theta "<<NNNvariance<<std::endl;
     avg_dec[u] = NNNaverage;
     var_dec_model[t][u] = NNNvariance;
  }//closing loop for nearest neighbor search
else
{
 var_dec_model[t][u] = -1;
}
 }//closing the for loop for varying radius values
 // std::cout<<"Average theta "<<avg_dec[0]<<std::endl;
 
}

//////////////////////////////////////////////Calculate Variance Based Descriptor for Scene Cloud///////////////////////////////////
 // Stores nearest neighbors in the cloud 
 // std::vector<int> pointIdxRadiusSearch;
 // std::vector<float> pointRadiusSquaredDistance;
  
 std::cout<<"Starting Scene Descriptor Calculation"<<std::endl; 
 radius = 0.5;//256.0f * rand () / (RAND_MAX + 1.0f);

 // float NNNaverage = 0;
 // float NNNvariance = 0;
  float var_dec_scene[scene_keypoints->size ()][3];
  std::cout<<"Size of scene cloud"<<scene_keypoints->size ()<<std::endl; 
for (int t = 0;t < (scene_keypoints->size ());t++)//
{ 

 float avg_dec[3];
       
 std::cout<<"t  "<<t<<std::endl;
 
 for(int u = 0;u<3;u++) // how many times radius for search is varied
 {
    // Neighbors within radius search
if ( kdtree.radiusSearch (scene_keypoints->at (t), radius*(u+1), pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
  {
  //Create parameters for cos-theta calculation 
  float Dot_prod[pointIdxRadiusSearch.size ()];
  float norm_vect[pointIdxRadiusSearch.size ()];
  float norm_keyp[pointIdxRadiusSearch.size ()];
  float theta[pointIdxRadiusSearch.size ()];
  
  std::cout<<"searched index "<< pointIdxRadiusSearch.size ()<<std::endl;
  
    for (size_t i = 0; i <pointIdxRadiusSearch.size (); ++i)//
    {    
     //Create vector to store normals of nearest neighbors
     //std::cout<<"normal value is "<<model_normals->at(pointIdxRadiusSearch[i]).normal_x<<std::endl;
    float norm_vect_x = scene_normals->at(pointIdxRadiusSearch[i]).normal_x;
    float norm_vect_y = scene_normals->at(pointIdxRadiusSearch[i]).normal_y;
    float norm_vect_z = scene_normals->at(pointIdxRadiusSearch[i]).normal_z;
    float keyp_vect_x = scene_normals->at(t).normal_x;
    float keyp_vect_y = scene_normals->at(t).normal_y;
    float keyp_vect_z = scene_normals->at(t).normal_z;
    //Components to calculate cos theta................Code not optimized
    Dot_prod[i]=(norm_vect_x*keyp_vect_x+norm_vect_y*keyp_vect_y+norm_vect_z*keyp_vect_z);
    norm_vect[i] = pow((norm_vect_x*norm_vect_x+norm_vect_y*norm_vect_y+norm_vect_z*norm_vect_z),0.5);
    norm_keyp[i] = pow((keyp_vect_x*keyp_vect_x+keyp_vect_y*keyp_vect_y+keyp_vect_z*keyp_vect_z),0.5);
    theta[i] = Dot_prod[i]/(norm_vect[i]*norm_keyp[i]);
    //std::cout<<" "<<theta[i]<<std::endl;
    
    //Calculate sum of all thetas to get average
     NNNaverage = NNNaverage + theta[i]; 
     }
     NNNaverage = NNNaverage/pointIdxRadiusSearch.size ();

     //Calculate variance 
      for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
    {
      NNNvariance = NNNvariance + pow((NNNaverage-theta[i]),2);
     }
     NNNvariance = NNNvariance/pointIdxRadiusSearch.size ();
    // std::cout<<"Average theta "<<NNNvariance<<std::endl;
     avg_dec[u] = NNNaverage;
     var_dec_scene[t][u] = NNNvariance;
  }//closing loop for nearest neighbor search
else
{
 var_dec_scene[t][u] = -1;
}
 }//closing the for loop for varying radius values
 // std::cout<<"Average theta "<<avg_dec[0]<<std::endl;
 
}
////////////////////////////////////////////////////////////End of Descriptor////////////////////////////////////////////////////
/////////////////////////////////////////////////////Write DEscriptor to File///////////////////////////////////////////////////
float Mar_desc ;
  ofstream myfile;
  myfile.open ("MarModel.txt");


  for (size_t i = 0; i < model_keypoints->size (); ++i)
  {
  for (size_t j = 0; j < 3; ++j)
  {
 //copyToFloatArray(model_descriptors,*shot_desc);
  //std::cout << "SHOT Descriptors for point: " << i<< "descriptor no: "<<j<<" value is "<<model_descriptors->at (i).descriptor[j]<< std::endl;
  myfile << i <<" "<<j<<" "<<var_dec_model[i][j]<< std::endl;
  }
  }
  myfile.close();
  myfile.open ("MarScene.txt");
  for (size_t i = 0; i < scene_keypoints->size (); ++i)
  {
  for (size_t j = 0; j < 3; ++j)
  {
 //copyToFloatArray(model_descriptors,*shot_desc);
  //std::cout << "SHOT Descriptors for point: " << i<< "descriptor no: "<<j<<" value is "<<model_descriptors->at (i).descriptor[j]<< std::endl;
  myfile << i <<" "<<j<<" "<<var_dec_scene[i][j]<< std::endl;
  }
  }
  myfile.close();
  
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/*
 pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud (model_descriptors);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.20f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
*/
  ////////////////////////////End of Routine ///////////////////////////////////////////////////////////////////////////////
   std::cout<<"exiting object recognition loop "<<std::endl;
   sub.shutdown();
   }
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
