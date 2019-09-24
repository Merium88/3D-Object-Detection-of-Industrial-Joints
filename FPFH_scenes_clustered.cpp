#include <iostream>
#include <string>
#include <sstream>
#include <math.h> 
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
#include <pcl/features/fpfh.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/region_growing.h>

using namespace std;
typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::FPFHSignature33 DescriptorType1;
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
float model_ss_ (0.02f);
float scene_ss_ (0.02f);
float rf_rad_ (0.02f);
float descr_rad_ (0.2f);
float cg_size_ (0.02f);
float cg_thresh_ (2.0f);//(2.0f);
//ros::Subscriber sub;
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

void cloud_cb ()
    {
      int num;
      pcl::PCDWriter writer;

     // Read scene pointcloud
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGBA>());
    
 ///////////////////////Read GRoundtruth text file///////////////////////

  std::vector<float> vec;
  std::vector<float> temp;
  ifstream myReadFile;
  int count = 0;
  float f[4];
  std::vector<vector <float> > pose;
 myReadFile.open("/home/umer/Documents/PCL-Workspace/stubcad/iros_data/pose.txt");
 char output[1000];
 if (myReadFile.is_open()) 
{
 while (!myReadFile.eof())
 {
    myReadFile >> output;
    //cout<<output<<endl;
    sscanf(output,"%f,%f,%f,%f\n",&f[0],&f[1],&f[2],&f[3]);
    //cout<<f[0]<<" "<<f[1]<<" "<<f[2]<<" "<<f[3]<<endl;
    for(int j=0;j<4;j++)
    vec.push_back(f[j]);
 }
}
myReadFile.close();
//cout<<(vec.size()-4)/12<<endl;
int n = 0;
for(int j=0;j<(vec.size()-4)/12;j++)
{
	pose.push_back(vector<float>());
	for(int k=0+count;k<12+count;k++)
	{
	 // cout<<vec[k]<<" ";
	  pose[j].push_back(vec[k]);
	  n++;
	}
	count = n;
}
/*
for(int k=0;k<pose[0].size();k++)
	{
	  cout<<pose[0][k]<<" ";
	}
cout<<endl;*/
    //////////////////////////////////////////////////////////////////Object Detection Module//////////////////////////////////////////////////// 
     //parseCommandLine (argc, argv);
   pcl::PointCloud<PointType>::Ptr full (new pcl::PointCloud<PointType> ());
   pcl::io::loadPCDFile ("/home/umer/Documents/PCL-Workspace/stubcad/New_partial/stubcad.pcd", *full);
   pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  //pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<PointType>::Ptr full_pose (new pcl::PointCloud<PointType> ());
 pcl::PointCloud<NormalType>::Ptr modelfilt_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scenefilt_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<PointType>::Ptr full_pose_t (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr full_pose_transform (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_ptr (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_full(new pcl::PointCloud<pcl::PointXYZRGBA>); 
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr translate_x(new pcl::PointCloud<pcl::PointXYZRGBA>); 
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr translate_y(new pcl::PointCloud<pcl::PointXYZRGBA>); 
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr full_constraint(new pcl::PointCloud<pcl::PointXYZRGBA>); 
 pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_model (new pcl::PointCloud<pcl::FPFHSignature33> ());
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_scene (new pcl::PointCloud<pcl::FPFHSignature33> ());
  scene_ptr = scene;

   string st1;
  for (int m=14;m<20;m++)
{   stringstream ss;
    std::cout << "Scene number " << m<<std::endl;
    string st3;int counter = 1;int counter2 = 1;string filename ;
    st3 = "/home/umer/Documents/PCL-Workspace/stubcad/iros_data/scene";
    ss << m;string st2 = ".pcd";string st4 = "_seg.pcd";
    filename = st3 + ss.str() + st4;
    int COUNTER = 0; int CORRS=0;
  if (pcl::io::loadPCDFile (filename, *scene) < 0)
     {
      std::cout << "Error loading scene cloud." << std::endl;
     }
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*scene,*scene, indices); 

   pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  //norm_est.setKSearch (20);
  norm_est.setRadiusSearch (0.05);
  pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
  norm_est.setSearchMethod(kdtree);
  norm_est.setInputCloud (scene);
  norm_est.compute (*scene_normals);
  pcl::UniformSampling<PointType> uniform_sampling;
 uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch (scene_ss_);
  uniform_sampling.filter (*scene_keypoints);
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
 norm_est.setInputCloud (scene_keypoints);
  norm_est.compute (*scenefilt_normals);
  pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud (scene_keypoints);
  fpfh.setInputNormals (scenefilt_normals);
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr treeFPFH_scene (new pcl::search::KdTree<pcl::PointXYZRGBA>);
  fpfh.setSearchMethod (treeFPFH_scene);
  fpfh.setRadiusSearch (0.15);
  fpfh.compute (*fpfh_scene);
  // fpfhs->points.size () should have the same size as the input cloud->points.size ()*
  
int cluster_grp;

  for (int l = 0; l < 3; ++l)
{
  for (int k=0;k<2;k++)
{
  //  Load CAD clouds
  //
if(k==0)
  st1 = "/home/umer/Documents/PCL-Workspace/stubcad/iros_data/chord";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd
  else if (k==1)
   st1 = "/home/umer/Documents/PCL-Workspace/stubcad/iros_data/stub";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd

  stringstream sm;
int h;
if (l==0)
    h = 18; //cluster 1
else if (l==1)
    h = 8;//cluster 2
else if (l==2)
    h = 0;//cluster 3

  sm << h;
  filename = st1 + sm.str() + st2;
  
    cout<<"Pose "<< l<< " Matched to Scene "<<m<<endl;
  if (pcl::io::loadPCDFile (filename, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    //showHelp (argv[0]);
    //return (-1);
  }
   st1 = "/home/umer/Documents/PCL-Workspace/stubcad/iros_data/";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd

  filename = st1 + sm.str() + st2;
  pcl::io::loadPCDFile (filename, *full_pose);
  
// REmove NaNs from POintcloud

  pcl::removeNaNFromPointCloud(*model,*model, indices); 
  pcl::removeNaNFromPointCloud(*full_pose,*full_pose, indices); 
  //  Compute Normals
  //
 //pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setInputCloud (model);
  //norm_est.setKSearch (20);
  norm_est.setRadiusSearch (0.05);

  norm_est.setSearchMethod(kdtree);
  norm_est.compute (*model_normals);

  //
  //  Downsample Clouds to Extract keypoints
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (0.001);
  uniform_sampling.filter (*model_keypoints);


  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
 

  //writer.write ("downsampled_only_chord_24112016.pcd", *model_keypoints, false);  
  //writer.write ("downsampled_scene_24112016.pcd", *scene_keypoints, false);  

   //  Compute Normals for downsampled scene
  // norm_est.setKSearch (20);
  norm_est.setRadiusSearch (0.05);
  norm_est.setSearchMethod(kdtree);
  norm_est.setInputCloud (model_keypoints);
  norm_est.compute (*modelfilt_normals);

 
 // Add another keypoint extractor
 pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
 
  //  Compute Descriptor for keypoints
  //
  if (use_SHOT_)
{
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);

 // Create the FPFH estimation class, and pass the input dataset+normals to it
  pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud (model_keypoints);
  fpfh.setInputNormals (modelfilt_normals);
  fpfh.setSearchMethod(tree);
 // cout<<model_keypoints->size()<<"    "<<model_normals->size()<<"  ";
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr treeFPFH (new pcl::search::KdTree<pcl::PointXYZRGBA>);
  fpfh.setSearchMethod (treeFPFH);
  fpfh.setRadiusSearch (0.15);
  fpfh.compute (*fpfh_model);
}

  //
  //  Find Model-Scene Correspondences with KdTree
  //
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

 
  pcl::KdTreeFLANN<DescriptorType1> match_search;
  match_search.setInputCloud (fpfh_model);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < fpfh_scene->points.size (); ++i)
  {
    std::vector<int> neigh_indices (2);
    std::vector<float> neigh_sqr_dists (2);
    if (!pcl_isfinite (fpfh_scene->points[i].histogram[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (fpfh_scene->points[i], 2, neigh_indices, neigh_sqr_dists);
    double tau = neigh_sqr_dists[0]/neigh_sqr_dists[1];
    if(found_neighs >= 1 && tau<=1 ) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {  
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;


  //  Actual Clustering
  //
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;

  //  Using Hough3D
  if (use_hough_)
 {
    //
    //  Compute (Keypoints) Reference Frames only for Hough
    //
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);  //vary and check

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_); //vary and check
    clusterer.setHoughThreshold (cg_thresh_); //vary and check
    clusterer.setUseInterpolation (false);
    clusterer.setUseDistanceWeight (true);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);
  }
  //  Output results
  //
 double prev_score; double prev_score2;double score2;double score;Eigen::Matrix4f Final_pose;
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
    CORRS = CORRS+rototranslations.size ();
 // cout<<"corrs "<<CORRS<<endl;
  if(rototranslations.size () >0)
  {
  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
   // std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
   // std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);
    Eigen::Matrix4f transformation_1 = Eigen::Matrix4f::Identity ();
    transformation_1.block<3,3>(0,0) = rotation;
    transformation_1.block<3,1>(0,3) = translation;

    //break the loop if transformation matrix is identity
     if(rotation(0,0) != 1) 
   {  
    ////////////////////////////////////////////////////////////////////////////////////////////// ICP Alignment /////////////////////////////////////////////////////////////////////
  
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
    // Calling ICP
  pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
  icp.setMaximumIterations (1);
  icp.setInputSource (rotated_model);
  icp.setInputTarget (scene);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_icp (new pcl::PointCloud<pcl::PointXYZRGBA>); 
  icp.align (*rotated_icp);
  score = icp.getFitnessScore();
  if(counter==1)
  {
      prev_score  = score;
  }
  counter++;
  printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
 if ((score < prev_score || score==prev_score) && counter>1 ) //&& score<0.00006
  {
    prev_score = score;
    icp_flag = false;
    cluster_grp = l+1;
    cout<<"cluster"<<cluster_grp<<endl;
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity ();
    transformation_matrix = icp.getFinalTransformation ().cast<float>();


  //////////////////////////////////////////////////////Load Partial Pose////////////////////////////////
  
  //////////Complete Transformation matrix is given feature transform*icp ttransform
    Final_pose = transformation_matrix*transformation_1;
    pcl::transformPointCloud (*full_pose, *full_pose_transform, Final_pose);
    icp.setMaximumIterations (1);
    icp.setInputSource (full_pose_transform);
    icp.setInputTarget (scene);
    icp.align (*rotated_full);
    //printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
    //score2 = icp.getFitnessScore(); 
    }
    std::stringstream ss_cloud;
    ss_cloud << "instance" << i;

   if (icp_flag==false)
    {
 COUNTER++;
  cout<<"No of matches found "<< COUNTER<<"POSE NO "<< l<<endl;
  //  Visualization
  //
  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
  viewer.addPointCloud (scene, "scene_cloud");
  viewer.setBackgroundColor (255, 255, 255);
  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

  if (show_correspondences_ || show_keypoints_)
  {
    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 200, 128);
   // viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
  }

  if (show_keypoints_)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
   // viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
   // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
  }

    // Plot icp rotated model    
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
    //viewer.addPointCloud (rotated_icp, rotated_model_color_handler, ss_cloud.str ());
    //PLot Partial view rotated after full transformation
    pcl::visualization::PointCloudColorHandlerCustom<PointType> full_pose_color_handler (full_pose_transform, 0, 255, 200);
    viewer.addPointCloud (full_pose_transform, full_pose_color_handler, "partial view keypoints");
    //PLot Partial view icp after full transformation
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_full_color_handler (rotated_icp, 150, 155, 200);
   // viewer.addPointCloud (rotated_full, rotated_full_color_handler, "partial view keypoints");
    
  while (!viewer.wasStopped ())
  {
    viewer.spin();
  }
   icp_flag = true;
    } //if score previous
    } 
    }
    }//if rototranslantation
    }
   }

///////////////////////////////////////////////////////////SEcond Layer of Tree//////////////////////////
//float model_ss_ (0.01f);
//float scene_ss_ (0.01f);
//float rf_rad_ (0.01f);
//float descr_rad_ (0.1f);
//float cg_size_ (0.02f);
//float cg_thresh_ (2.0f);//(2.0f);



cout<<"Cluster group choosen is "<<cluster_grp<<endl;
int next_size;
CORRS=0;COUNTER = 0;
if(cluster_grp == 1)
   next_size = 6;
else if(cluster_grp == 2)
   next_size = 13;
else if(cluster_grp == 3)
   next_size = 14;


string st5;

  for (int l = 0; l < 42; ++l)
{
  for (int k=0;k<2;k++)
{
  //  Load CAD clouds
  stringstream cl;
if(k==0)
{
  st1 = "/home/umer/Documents/PCL-Workspace/stubcad/iros_data";///home/umer/Documents/PCL-Workspace/stubcad/iros_data";///home/nus/catkin_ws/stub_poses/3.pcd   
 cl<<cluster_grp;
  st5 = "/chord";
}
  else if (k==1)
{
   st1 = "/home/umer/Documents/PCL-Workspace/stubcad/iros_data";///home/umer/Documents/PCL-Workspace/stubcad/iros_data";///home/umer/Documents/PCL-Workspace/stubcad/iros_data/cluster";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd
   st5 = "/stub";
    cl<<cluster_grp;
}
  stringstream sm;

  sm << l;
  filename = st1+st5+sm.str() + st2;// +cl.str()
    cout<<"Pose "<< l<< " Matched to Scene "<<m<<endl;
  if (pcl::io::loadPCDFile (filename, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    //showHelp (argv[0]);
    //return (-1);
  }
   st1 = "/home/umer/Documents/PCL-Workspace/stubcad/iros_data/";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd

  filename = st1 + sm.str() + st2;
  pcl::io::loadPCDFile (filename, *full_pose);
  

// REmove NaNs from POintcloud
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*model,*model, indices); 
  pcl::removeNaNFromPointCloud(*full_pose,*full_pose, indices); 
  //  Compute Normals
  //
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setInputCloud (model);
  //norm_est.setKSearch (20);
  norm_est.setRadiusSearch (0.05);
  pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
  norm_est.setSearchMethod(kdtree);
  norm_est.compute (*model_normals);
  //
  //  Downsample Clouds to Extract keypoints
  //

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.filter (*model_keypoints);

  // model_keypoints = model;
  //scene_keypoints = scene;

  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
   //  Compute Normals for downsampled scene
  // norm_est.setKSearch (20);
  norm_est.setRadiusSearch (0.05);
  norm_est.setSearchMethod(kdtree);
  norm_est.setInputCloud (model_keypoints);
  norm_est.compute (*modelfilt_normals);
 // Add another keypoint extractor
 pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
 
  //  Compute Descriptor for keypoints
  //
  if (use_SHOT_)
{
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);

 // Create the FPFH estimation class, and pass the input dataset+normals to it
  pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud (model_keypoints);
  fpfh.setInputNormals (modelfilt_normals);
  fpfh.setSearchMethod(tree);
 // cout<<model_keypoints->size()<<"    "<<model_normals->size()<<"  ";
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr treeFPFH (new pcl::search::KdTree<pcl::PointXYZRGBA>);
  fpfh.setSearchMethod (treeFPFH);
  fpfh.setRadiusSearch (0.15);
  fpfh.compute (*fpfh_model);
  
}

  //
  //  Find Model-Scene Correspondences with KdTree
  //
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
  pcl::KdTreeFLANN<DescriptorType1> match_search;
  match_search.setInputCloud (fpfh_model);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < fpfh_scene->points.size (); ++i)
  {
    std::vector<int> neigh_indices (2);
    std::vector<float> neigh_sqr_dists (2);
    if (!pcl_isfinite (fpfh_scene->points[i].histogram[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (fpfh_scene->points[i], 2, neigh_indices, neigh_sqr_dists);
    double tau = neigh_sqr_dists[0]/neigh_sqr_dists[1];
    if(found_neighs >= 1  && tau<=1) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {  
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }
  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;


  //  Actual Clustering
  //
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;

  //  Using Hough3D

    //
    //  Compute (Keypoints) Reference Frames only for Hough
    //
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);  //vary and check

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_); //vary and check
    clusterer.setHoughThreshold (cg_thresh_); //vary and check
    clusterer.setUseInterpolation (false);
    clusterer.setUseDistanceWeight (true);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);

  //
  //  Output results
  //
 double prev_score=0; double prev_score2=0;double score2=0;double score=0;Eigen::Matrix4f Final_pose;
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
    CORRS = CORRS+rototranslations.size ();
  cout<<"corrs "<<CORRS<<endl;
  if(rototranslations.size () >0)
  {
  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
   // std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
   // std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);
    Eigen::Matrix4f transformation_1 = Eigen::Matrix4f::Identity ();
    transformation_1.block<3,3>(0,0) = rotation;
    transformation_1.block<3,1>(0,3) = translation;

    //break the loop if transformation matrix is identity
     if(rotation(0,0) != 1) 
   {  
    ////////////////////////////////////////////////////////////////////////////////////////////// ICP Alignment /////////////////////////////////////////////////////////////////////
  
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
    // Calling ICP
  pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
  icp.setMaximumIterations (1);
  icp.setInputSource (rotated_model);
  icp.setInputTarget (scene);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_icp (new pcl::PointCloud<pcl::PointXYZRGBA>); 
  icp.align (*rotated_icp);
  score = icp.getFitnessScore();
  if(counter==1)
  {
      prev_score  = score;
  }
  counter++;
  //printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
  //if ((score < prev_score || score==prev_score) && counter>1 ) //&& score<0.00006
  {
    prev_score = score;
   // icp_flag = false;
      Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity ();
    transformation_matrix = icp.getFinalTransformation ().cast<float>();


  //////////////////////////////////////////////////////Load Partial Pose///////////////////////////////////////
  
  //////////Complete Transformation matrix is given feature transform*icp ttransform
    Final_pose = transformation_matrix*transformation_1;
    pcl::transformPointCloud (*full_pose, *full_pose_transform, Final_pose);
    icp.setMaximumIterations (1);
    icp.setInputSource (full_pose_transform);
    icp.setInputTarget (scene);
    icp.align (*rotated_full);
    //printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
    //score2 = icp.getFitnessScore(); 
    }
    std::stringstream ss_cloud;
    ss_cloud << "instance" << i;
    
   //////////////////////////////////////////////////////Load Full Pose///////////////////////////////////////
    Eigen::Matrix4f transformation_matrix2 = Eigen::Matrix4f::Identity ();
    transformation_matrix2 = icp.getFinalTransformation ().cast<float>();
    Eigen::Matrix4f T_org = Eigen::Matrix4f::Identity (); //load transformation of CAD to partial view
    T_org(0,0)=pose[l][0];
    T_org(0,1)= pose[l][1];
    T_org(0,2)= pose[l][2];
    T_org(0,3)= pose[l][3];
    T_org(1,0)= pose[l][4]; 
    T_org(1,1)= pose[l][5];
    T_org(1,2)= pose[l][6];
    T_org(1,3)= pose[l][7];
    T_org(2,0)= pose[l][8];
    T_org(2,1)= pose[l][9];
    T_org(2,2)= pose[l][10];
    T_org(2,3)= pose[l][11];
    Eigen::Matrix4f Full_pose_new = transformation_matrix2*Final_pose*T_org;
    pcl::transformPointCloud (*full, *full_pose_t, Full_pose_new);
    icp.setMaximumIterations (1);
    icp.setInputSource (full_pose_t);
    icp.setInputTarget (scene);
    icp.align (*rotated_full);
    printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
    score2 = icp.getFitnessScore(); 
    if(counter2==1)
    {
	prev_score2  = score2;
    }
     counter2++;
    //if ((score2 < prev_score2 || score2==prev_score2) && counter2>1) //&& score<0.00006
    {
     prev_score2 = score2;
     icp_flag = false;
   
 printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Full_pose_new (0,0), Full_pose_new(0,1), Full_pose_new (0,2),Full_pose_new(0,3));
    printf ("        R = | %6.3f %6.3f %6.3f %6.3f | \n", Full_pose_new (1,0), Full_pose_new(1,1), Full_pose_new(1,2),Full_pose_new(1,3));
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Full_pose_new (2,0), Full_pose_new (2,1), Full_pose_new (2,2),Full_pose_new(2,3));
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Full_pose_new (3,0), Full_pose_new (3,1), Full_pose_new (3,2),Full_pose_new(3,3));
    printf ("\n");
     } 

    //transformation_matrix3 = icp.getFinalTransformation ().cast<float>();
    //Eigen::Matrix4f Full_pose_finale = transformation_matrix3*Full_pose_new;
    /////////// Show bounding box/////////////////////////
  pcl::PointCloud<pcl::PointXYZ>::Ptr bounding_box(new pcl::PointCloud<pcl::PointXYZ>);
bounding_box->resize(rotated_full->size());

for (size_t i = 0; i < rotated_full->size(); i++) {
    bounding_box->at(i).x = rotated_full->at(i).x;
    bounding_box->at(i).y = rotated_full->at(i).y;
    bounding_box->at(i).z = rotated_full->at(i).z;
}

//Crop stubcad root node to create bounding box
  pcl::ExtractIndices<pcl::PointXYZ> extractn;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);
  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (bounding_box);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (1000);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (bounding_box);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (10.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (10.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  for(int j = 0;j<clusters.size();j++)
  {

  pcl::IndicesPtr indices_ptr (new std::vector<int> (clusters[j].indices.size ())); 
  for (int i = 0; i < indices_ptr->size (); i++) 
          (*indices_ptr)[i] = clusters[j].indices[i]; 

  extractn.setInputCloud (bounding_box);
  extractn.setIndices (indices_ptr);
  extractn.setNegative (false);
if(j==0)
{
  extractn.filter (*cloud_plane);
  //viewer.addPointCloud(cloud_plane);
}
}
 
// compute principal direction
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_plane, centroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*cloud_plane, centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
    eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

    // move the points to the that reference frame
    Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
    p2w.block<3,3>(0,0) = eigDx.transpose();
    p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * centroid.head<3>());
    pcl::PointCloud<pcl::PointXYZ> cPoints;
    pcl::transformPointCloud(*cloud_plane, cPoints, p2w);

    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(cPoints, min_pt, max_pt);
    const Eigen::Vector3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());

    // final transform
    const Eigen::Quaternionf qfinal(eigDx);
    const Eigen::Vector3f tfinal = eigDx*mean_diag + centroid.head<3>();

  cout<<"centroid x "<<centroid[0]<<" y "<<centroid[1]<<" z "<<centroid[2]<<endl;
  cout<<"max vertices x "<<max_pt.x<<" y "<<max_pt.y<<" z "<<max_pt.z<<endl;
  cout<<"min vertices x "<<min_pt.x<<" y "<<min_pt.y<<" z "<<min_pt.z<<endl;
  cout<<"translation "<<tfinal[0]<<"  "<<tfinal[1]<<"  "<<tfinal[2]<<endl;

  Eigen::Vector3f retVector;

    float x = qfinal.y();
    float y = qfinal.z();
    float z = qfinal.x();
    float w = qfinal.w();

    retVector[0] = atan2(2.0 * (y * z + w * x), w * w - x * x - y * y + z * z);
    retVector[1] = asin(-2.0 * (x * z - w * y));
    retVector[2] = atan2(2.0 * (x * y + w * z), w * w + x * x - y * y - z * z);

    retVector[0] = (retVector[0] * (180 / M_PI));
    if(abs(retVector[0])>90)
    {
    if(retVector[0]<0)
     retVector[0] = -180-retVector[0];
 
    if(retVector[0]>0)
     retVector[0] = 180-retVector[0];
    }
    retVector[1] = (retVector[1] * (180 / M_PI))*-1;
    if(abs(retVector[1])>90)
    {
    if(retVector[1]<0)
     retVector[1] = -180-retVector[1];
 
    if(retVector[1]>0)
     retVector[1] = 180-retVector[1];
    }
    retVector[2] = retVector[2] * (180 / M_PI);
  cout<<"rotation "<<retVector[0]<<"  "<<retVector[1]<<"  "<<retVector[2]<<endl;
 
    Eigen::Matrix4f x_constraint = Eigen::Matrix4f::Identity ();
    Eigen::Matrix4f y_constraint = Eigen::Matrix4f::Identity ();
    Eigen::Matrix4f move_to_origin = Eigen::Matrix4f::Identity ();
    x_constraint(1,1) = cos(-180*M_PI/180);
    x_constraint(1,2) = -sin(-180*M_PI/180);
    x_constraint(2,1) = sin(-180*M_PI/180);
    x_constraint(2,2) = cos(-180*M_PI/180);
    y_constraint(0,0) = cos(retVector[1]*M_PI/180);
    y_constraint(0,2) = sin(retVector[1]*M_PI/180);
    y_constraint(2,0) = -sin(retVector[1]*M_PI/180);
    y_constraint(2,2) = cos(retVector[1]*M_PI/180);
    //centroid of transformed pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr origin (new pcl::PointCloud<pcl::PointXYZ> ());
    origin->resize(rotated_full->size());

    for (size_t i = 0; i < rotated_full->size(); i++) {
    origin->at(i).x = rotated_full->at(i).x;
    origin->at(i).y = rotated_full->at(i).y;
    origin->at(i).z = rotated_full->at(i).z;
    }
    Eigen::Vector4f centroidnew;
    pcl::compute3DCentroid(*origin, centroidnew);
    cout<<centroidnew[0] << endl << centroidnew[1] << endl << centroidnew[2] << endl; 
    move_to_origin(0,3) = -centroidnew[0];
    move_to_origin(1,3) = -centroidnew[1];
    move_to_origin(2,3) = -centroidnew[2];
   // pcl::transformPointCloud (*rotated_full, *translate_x, move_to_origin);
   // pcl::transformPointCloud (*translate_x, *full_constraint, x_constraint);
    //pcl::transformPointCloud (*full_constraint, *translate_y, y_constraint);
    move_to_origin(0,3) = centroidnew[0];
    move_to_origin(1,3) = centroidnew[1];
    move_to_origin(2,3) = centroidnew[2];
    //pcl::transformPointCloud (*full_constraint,*translate_y, move_to_origin);

// Update bounding box location
 /* 
pcl::PointCloud<pcl::PointXYZ>::Ptr bounding_box_new(new pcl::PointCloud<pcl::PointXYZ>);
bounding_box_new->resize(translate_y->size());
for (size_t i = 0; i < translate_y->size(); i++) {
    bounding_box_new->at(i).x = translate_y->at(i).x;
    bounding_box_new->at(i).y = translate_y->at(i).y;
    bounding_box_new->at(i).z = translate_y->at(i).z;
}

//Crop stubcad root node to create bounding box
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (bounding_box_new);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);
  reg.setMinClusterSize (1000);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (bounding_box_new);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (10.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (10.0);
  reg.extract (clusters);
  for(int j = 0;j<clusters.size();j++)
  {

  pcl::IndicesPtr indices_ptr (new std::vector<int> (clusters[j].indices.size ())); 
  for (int i = 0; i < indices_ptr->size (); i++) 
          (*indices_ptr)[i] = clusters[j].indices[i]; 

  extractn.setInputCloud (bounding_box_new);
  extractn.setIndices (indices_ptr);
  extractn.setNegative (false);
if(j==0)
{
  extractn.filter (*cloud_plane);
  //viewer.addPointCloud(cloud_plane);
}
}
 
// compute principal direction
    pcl::compute3DCentroid(*cloud_plane, centroid);
    computeCovarianceMatrixNormalized(*cloud_plane, centroid, covariance);
    eigDx = eigen_solver.eigenvectors();
    eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

    // move the points to the that reference frame
    p2w.block<3,3>(0,0) = eigDx.transpose();
    p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * centroid.head<3>());
    pcl::transformPointCloud(*cloud_plane, cPoints, p2w);

    pcl::PointXYZ min_pt_update, max_pt_update;
    pcl::getMinMax3D(cPoints, min_pt_update, max_pt_update);
    const Eigen::Vector3f mean_diag_update = 0.5f*(max_pt_update.getVector3fMap() + min_pt_update.getVector3fMap());

    // final transform
    const Eigen::Quaternionf qfinal_update(eigDx);
    const Eigen::Vector3f tfinal_update = eigDx*mean_diag_update + centroid.head<3>();

  cout<<"new max vertices x "<<max_pt_update.x<<" y "<<max_pt_update.y<<" z "<<max_pt_update.z<<endl;
  cout<<"new min vertices x "<<min_pt_update.x<<" y "<<min_pt_update.y<<" z "<<min_pt_update.z<<endl;
  cout<<"new translation "<<tfinal_update[0]<<"  "<<tfinal_update[1]<<"  "<<tfinal_update[2]<<endl; */
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // if (icp_flag==false)
    {
 COUNTER++;
  cout<<"No of matches found "<< COUNTER<<"POSE NO "<< l<<endl;

  
  //  Visualization
  //
  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
  viewer.addPointCloud (scene, "scene_cloud");
  viewer.setBackgroundColor (250, 250, 250);
  //viewer.addCoordinateSystem (0.5, "rotated_full", 0); 
  viewer.addCoordinateSystem (0.5); 
  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

  if (show_correspondences_ || show_keypoints_)
  {
    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 200, 128);
   // viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
  }

  if (show_keypoints_)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
   // viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
   // viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
  }

    // Plot icp rotated model    
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
    //viewer.addPointCloud (rotated_icp, rotated_model_color_handler, ss_cloud.str ());
    //PLot Partial view rotated after full transformation-green is after constraint
    //pcl::visualization::PointCloudColorHandlerCustom<PointType> full_pose_color_handler (translate_y, 0, 255, 200);
    //viewer.addPointCloud (translate_y, full_pose_color_handler, "partial view keypoints");
     //pcl::visualization::PointCloudColorHandlerCustom<PointType> full_pose_color_const (full_constraint, 255, 200, 0);
    //viewer.addPointCloud (full_constraint, full_pose_color_const, "at origin");
    //PLot Partial view icp after full transformation-blue is original
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_full_color_handler (rotated_full, 50, 155, 200);
    viewer.addPointCloud (rotated_full, rotated_full_color_handler, "rotated view keypoints");

    if (show_correspondences_)
    {
      for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
      {
        std::stringstream ss_line;
        ss_line << "correspondence_line" << i << "_" << j;
        PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
        PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        //viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
      }
    }

   // draw the cloud and the box
    //viewer.addPointCloud(point_cloud_ptr);
    viewer.addCube(tfinal, qfinal, max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z,"cube", 0);


  while (!viewer.wasStopped ())
  {
    viewer.spin();
  }
   icp_flag = true;
    } //if score previous
    } 
    }
    }//if rototranslantation
    }
   }

///////////////////////////////////////////End of Second Layer////////////////////////////////////////////



   }//End of loop for a scene
   std::cout<<"exiting object recognition loop "<<std::endl;
   //sub.shutdown();
   }

int main (int argc, char** argv)
   {
     cloud_cb();  
   }
