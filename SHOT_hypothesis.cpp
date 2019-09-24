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
//#include </home/nus/catkin_ws/src/my_pcl_tutorial/src/render_views_tesselated_sphere.h>
#include <pcl/recognition/hv/hv_go.h>

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
float model_ss_ (0.005f);
float scene_ss_ (0.01f);
float rf_rad_ (0.01f);
float descr_rad_ (0.01f);
float cg_size_ (0.02f);
float cg_thresh_ (2.0f);//(2.0f);
int icp_max_iter_ (1);
float icp_corr_distance_ (0.001f);
float hv_clutter_reg_ (0.001f);
float hv_inlier_th_ (0.005f);
float hv_occlusion_th_ (0.001f);
float hv_rad_clutter_ (0.003f);
float hv_regularizer_ (0.001f);
float hv_rad_normals_ (0.005);
bool hv_detect_clutter_ (false);


ros::Subscriber sub;
struct CloudStyle
{
    double r;
    double g;
    double b;
    double size;

    CloudStyle (double r,
                double g,
                double b,
                double size) :
        r (r),
        g (g),
        b (b),
        size (size)
    {
    }
};

CloudStyle style_white (255.0, 255.0, 255.0, 4.0);
CloudStyle style_red (255.0, 0.0, 0.0, 3.0);
CloudStyle style_green (0.0, 255.0, 0.0, 5.0);
CloudStyle style_cyan (93.0, 200.0, 217.0, 4.0);
CloudStyle style_violet (255.0, 0.0, 255.0, 8.0);


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
      cout<<"inside"<<endl;
     // Read scene pointcloud
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGBA>());
    
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
  pcl::PointCloud<PointType>::Ptr scene_ptr (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_full(new pcl::PointCloud<pcl::PointXYZRGBA>); 
  scene_ptr = scene;
  string st1;
  string st3;
  for (int m=0;m<11;m++)
{   std::cout << "Scene number " << m<<std::endl;
  for (int k=0;k<2;k++)
{  std::cout << "stub or chord " << k<<std::endl;
  for (int l = 0; l < 4; ++l)
{  std::cout << "which partial " << l<<std::endl;
  
  //  Load CAD clouds
   if(k==0)
   st1 = "/home/nus/catkin_ws/stub_partial/only_chord_";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd
   else if (k==1)
   st1 = "/home/nus/catkin_ws/stub_partial/only_stub_";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd

  // st1 = "/home/nus/catkin_ws/stub_poses/";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd
//string st1 = "/home/nus/catkin_ws/stub_partial/only_chord_";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd
string st2 = ".pcd";
string filename ;
int h = 0;

  stringstream sm;
    stringstream ss;
if (l==0)
    h = 3;
else if (l==1)
    h = 5;
else if (l==2)
    h = 8;
else if (l==3)
    h = 18;

  sm << h;
  filename = st1 + sm.str() + st2;
  
    cout<<"Pose "<< h<< " Matched to Scene "<<endl;
  if (pcl::io::loadPCDFile (filename, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    //showHelp (argv[0]);
    //return (-1);
  }
   st1 = "/home/nus/catkin_ws/stub_poses/";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd

  filename = st1 + sm.str() + st2;
  pcl::io::loadPCDFile (filename, *full_pose);
  
  st3 = "/home/nus/catkin_ws/Workshop_scene/scene";
  ss << m;
  filename = st3 + ss.str() + st2;
  if (pcl::io::loadPCDFile (filename, *scene) < 0)
     {
      std::cout << "Error loading scene cloud." << std::endl;
     }
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
  pcl::removeNaNFromPointCloud(*full_pose,*full_pose, indices); 
  //  Compute Normals
  //
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (20);
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
 /*  if (pcl::io::loadPCDFile ("only_chord_8.pcd", *model_keypoints) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    //showHelp (argv[0]);
    //return (-1);
  }*/
  // model_keypoints = model;
  //scene_keypoints = scene;

  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
  uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch (scene_ss_);
  uniform_sampling.filter (*scene_keypoints);
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

  //writer.write ("downsampled_only_chord_24112016.pcd", *model_keypoints, false);  
  //writer.write ("downsampled_scene_24112016.pcd", *scene_keypoints, false);  

 
 // Add another keypoint extractor
 pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
 
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
  
}

  //
  //  Find Model-Scene Correspondences with KdTree
  //
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud (model_descriptors);
  std::vector<int> model_good_keypoints_indices;
  std::vector<int> scene_good_keypoints_indices;
  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0]))  //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
      model_good_keypoints_indices.push_back (corr.index_query);
      scene_good_keypoints_indices.push_back (corr.index_match);
    }
  }
  pcl::PointCloud<PointType>::Ptr model_good_kp (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_good_kp (new pcl::PointCloud<PointType> ());
  pcl::copyPointCloud (*model_keypoints, model_good_keypoints_indices, *model_good_kp);
  pcl::copyPointCloud (*scene_keypoints, scene_good_keypoints_indices, *scene_good_kp);

  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

  //
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
  else // Using GeometricConsistency
  {
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize (cg_size_);
    gc_clusterer.setGCThreshold (cg_thresh_);

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize (rototranslations, clustered_corrs);
  }

  //
  //  Output results
  //
  int counter = 1;double prev_score; double prev_score2;int counter2 = 1;double score2;double score;
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
  std::vector<pcl::PointCloud<PointType>::ConstPtr> instances;
  std::vector<pcl::PointCloud<PointType>::ConstPtr> registered_instances;
  if(rototranslations.size () >0)
  {
  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    //std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
   // std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);
    Eigen::Matrix4f transformation_1 = Eigen::Matrix4f::Identity ();
    transformation_1.block<3,3>(0,0) = rotation;
    transformation_1.block<3,1>(0,3) = translation;
/*
    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  */  
    //break the loop if transformation matrix is identity
   if(rotation(0,0) != 1) 
   {  
    ////////////////////////////////////////////////////////////////////////////////////////////// ICP Alignment /////////////////////////////////////////////////////////////////////

    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
    instances.push_back (rotated_model);
    // Calling ICP
  pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
  icp.setMaximumIterations (100);
  icp.setMaxCorrespondenceDistance (icp_corr_distance_);
  icp.setInputSource (rotated_model);
  icp.setInputTarget (scene);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_icp (new pcl::PointCloud<pcl::PointXYZRGBA>); 
  icp.align (*rotated_icp);
  //registered_instances.push_back (rotated_icp);
  score = icp.getFitnessScore();
     
  if(counter==1)
  {
      prev_score  = score;
  }
  counter++;
  //printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
  if ((score < prev_score || score==prev_score) && counter>1 ) //&& score<0.00006
  {
    prev_score = score;
   // icp_flag = false;
      Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity ();
    transformation_matrix = icp.getFinalTransformation ().cast<float>();

  //////////////////////////////////////////////////////////////////Compute TRansformations for TCP Here///////////////////////////////////////////////////////////////////////////
  
  //////////Complete Transformation matrix is given feature transform*icp ttransform
  Eigen::Matrix4f Final_pose = transformation_matrix*transformation_1;
    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Final_pose (0,0), Final_pose(0,1), Final_pose (0,2),Final_pose(0,3));
    printf ("        R = | %6.3f %6.3f %6.3f %6.3f | \n", Final_pose (1,0), Final_pose (1,1), Final_pose (1,2),Final_pose(1,3));
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Final_pose (2,0), Final_pose (2,1), Final_pose (2,2),Final_pose(2,3));
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Final_pose (3,0), Final_pose (3,1), Final_pose (3,2),Final_pose(3,3));
    printf ("\n");

  //////////////////////////////////////////////////////Load Full Pose///////////////////////////////////////
    

    pcl::transformPointCloud (*full_pose, *full_pose_transform, Final_pose);
    icp.setMaximumIterations (icp_max_iter_);
    icp.setMaxCorrespondenceDistance (icp_corr_distance_);
    icp.setInputSource (full_pose_transform);
    icp.setInputTarget (scene);
    icp.align (*rotated_full);
    registered_instances.push_back (rotated_full);
    //printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
    score2 = icp.getFitnessScore(); 
      
		if(counter2==1)
	    {
		  prev_score2  = score2;
	    }
	    counter2++;
       //if ((score2 < prev_score2 || score2==prev_score2) && counter2>1) //&& score<0.00006
		//{
		//prev_score2 = score2;
		icp_flag = false;
		//} 
    }
  
  std::stringstream ss_cloud;
  ss_cloud << "instance" << i;
/*  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
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
    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
  }

  if (show_keypoints_)
  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
  }

    // Plot icp rotated model    
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
    viewer.addPointCloud (rotated_icp, rotated_model_color_handler, ss_cloud.str ());
    //PLot Partial view rotated after full transformation
    pcl::visualization::PointCloudColorHandlerCustom<PointType> full_pose_color_handler (full_pose, 0, 255, 200);
    viewer.addPointCloud (full_pose_transform, full_pose_color_handler, "partial view keypoints");
    //PLot Partial view icp after full transformation
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_full_color_handler (rotated_full, 150, 155, 200);
    viewer.addPointCloud (rotated_full, rotated_full_color_handler, "partial view keypoints");
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

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce();
  }*/
      // std::cout << "reached here visualization"<<endl;
     }
    }
    }
    
     /**
   * Hypothesis Verification
   */
  if (registered_instances.size ()>0)
  {
  cout << "--- Hypotheses Verification ---" << endl;
  std::vector<bool> hypotheses_mask;  // Mask Vector to identify positive hypotheses

  pcl::GlobalHypothesesVerification<PointType, PointType> GoHv;

  GoHv.setSceneCloud (scene);  // Scene Cloud
  GoHv.addModels (registered_instances, true);  //Models to verify

  GoHv.setInlierThreshold (hv_inlier_th_);
  GoHv.setOcclusionThreshold (hv_occlusion_th_);
  GoHv.setRegularizer (hv_regularizer_);
  GoHv.setRadiusClutter (hv_rad_clutter_);
  GoHv.setClutterRegularizer (hv_clutter_reg_);
  GoHv.setDetectClutter (hv_detect_clutter_);
  GoHv.setRadiusNormals (hv_rad_normals_);

  GoHv.verify ();
  GoHv.getMask (hypotheses_mask);  // i-element TRUE if hvModels[i] verifies hypotheses

  for (int i = 0; i < hypotheses_mask.size (); i++)
  {
    if (hypotheses_mask[i])
    {
      cout << "Instance " << i << " is GOOD! <---" << endl;
      //if (icp_flag==false)
   // {
  pcl::visualization::PCLVisualizer viewer ("Hypotheses Verification");
  viewer.addPointCloud (scene, "scene_cloud");

  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

  pcl::PointCloud<PointType>::Ptr off_model_good_kp (new pcl::PointCloud<PointType> ());
  pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));
  pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));
  pcl::transformPointCloud (*model_good_kp, *off_model_good_kp, Eigen::Vector3f (-1, 0, 0), Eigen::Quaternionf (1, 0, 0, 0));

  if (show_keypoints_)
  {
    CloudStyle modelStyle = style_white;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, modelStyle.r, modelStyle.g, modelStyle.b);
    viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, modelStyle.size, "off_scene_model");
  }

  if (show_keypoints_)
  {
    CloudStyle goodKeypointStyle = style_violet;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> model_good_keypoints_color_handler (off_model_good_kp, goodKeypointStyle.r, goodKeypointStyle.g,
                                                                                                    goodKeypointStyle.b);
    viewer.addPointCloud (off_model_good_kp, model_good_keypoints_color_handler, "model_good_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "model_good_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_good_keypoints_color_handler (scene_good_kp, goodKeypointStyle.r, goodKeypointStyle.g,
                                                                                                    goodKeypointStyle.b);
    viewer.addPointCloud (scene_good_kp, scene_good_keypoints_color_handler, "scene_good_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "scene_good_keypoints");
  }

 // for (size_t i = 0; i < instances.size (); ++i)
  //{
    std::stringstream ss_instance;
    ss_instance << "instance_" << i;

    CloudStyle clusterStyle = style_red;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> instance_color_handler (instances[i], clusterStyle.r, clusterStyle.g, clusterStyle.b);
    //viewer.addPointCloud (instances[i], instance_color_handler, ss_instance.str ());
    //viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, clusterStyle.size, ss_instance.str ());

    CloudStyle registeredStyles = hypotheses_mask[i] ? style_green : style_cyan;
    ss_instance << "_registered" << endl;
    pcl::visualization::PointCloudColorHandlerCustom<PointType> registered_instance_color_handler (registered_instances[i], registeredStyles.r,
                                                                                                   registeredStyles.g, registeredStyles.b);
    viewer.addPointCloud (registered_instances[i], registered_instance_color_handler, ss_instance.str ());
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, registeredStyles.size, ss_instance.str ());
  //}

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }
   //icp_flag = true;
    }
    else
    {
      cout << "Instance " << i << " is bad!" << endl;
    }
  }
  cout << "-------------------------------" << endl;
    }
    }//exiting l for loop
    }//exiting k for loop
    }//exiting m for loop
   std::cout<<"exiting object recognition loop "<<std::endl;
   sub.shutdown();
   }

int main (int argc, char** argv)
   {
     // Initialize ROS
     ros::init (argc, argv, "my_pcl_tutorial");
     ros::NodeHandle nh;  
     // Create a ROS subscriber for the input point cloud "depth"
     //sub = nh.subscribe ("/camera/depth_registered/points", 1, cloud_cb);
     cloud_cb();
     // Spin
     ros::spin ();
   
   }
