#include <iostream>
#include <string>
#include <sstream>
#include <pcl/io/pcd_io.h>
#include <geometry_msgs/Vector3.h>
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
#include <pcl_ros/point_cloud.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;
typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::FPFHSignature33 DescriptorType1;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> ColorHandlerT;
std::string model_filename_;
std::string scene_filename_;
ros::Subscriber sub;

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
float descr_rad_ (0.02f);
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

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
       int num;
      pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
      // Container for original & filtered data
      pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2; 
      pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
      pcl::PCLPointCloud2 cloud_filtered;
      pcl::PointCloud<pcl::PointXYZRGBA> cloudvrep;

 
     // Convert to PCL data type
     pcl_conversions::toPCL(*cloud_msg, *cloud);
     pcl::fromROSMsg(*cloud_msg, cloudvrep);
     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGBA>(cloudvrep));

      pcl::PCDWriter writer;

     // Read scene pointcloud
    // pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGBA>());
    
 ///////////////////////////////////////////////////////////////////////////////////Read GRoundtruth text file////////////////////////////////////////////////////////

  std::vector<float> vec;
  std::vector<float> temp;
  ifstream myReadFile;
  int count = 0;
  float f[4];
  std::vector<vector <float> > pose;
 myReadFile.open("/home/nus/catkin_ws/iros_data/pose.txt");
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

/////////////////////////////////////////////////////////////////////////Object Detection Module//////////////////////////////////////////////////// 
     //parseCommandLine (argc, argv);
   pcl::PointCloud<PointType>::Ptr full (new pcl::PointCloud<PointType> ());
   pcl::PointCloud<PointType>::Ptr down_full (new pcl::PointCloud<PointType> ());
   pcl::io::loadPCDFile ("/home/nus/catkin_ws/iros_data/stubcad.pcd", *full);
  pcl::UniformSampling<PointType> uniform_sampling;
 uniform_sampling.setInputCloud (full);
  uniform_sampling.setRadiusSearch (0.01);
  uniform_sampling.filter (*down_full);
   pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  //pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType1>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType1> ());
  pcl::PointCloud<DescriptorType1>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType1> ());
  pcl::PointCloud<PointType>::Ptr full_pose (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr modelfilt_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scenefilt_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<PointType>::Ptr full_pose_t (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr full_pose_transform (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_ptr (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_full(new pcl::PointCloud<pcl::PointXYZRGBA>); 
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_model (new pcl::PointCloud<pcl::FPFHSignature33> ());
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_scene (new pcl::PointCloud<pcl::FPFHSignature33> ());
  pcl::PointCloud<PointType>::Ptr scene_crop(new pcl::PointCloud<PointType>);
  scene_ptr = scene;

   string st1;
 int m=1;
{   stringstream ss;
    //std::cout << "Scene number " << m<<std::endl;
    string st3;int counter = 1;int counter2 = 1;string filename ;
    ss << m;string st2 = ".pcd";string st4 = ".pcd";
    int COUNTER = 0; int CORRS=0;

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*scene,*scene, indices); 
 
//////////////////////////////////////////////////////////////////////////////////////Segment Scene//////////////////////////////////////////////////
pcl::PointCloud<pcl::PointXYZ>::Ptr scene_new(new pcl::PointCloud<pcl::PointXYZ>);
scene_new->resize(scene->size());
for (size_t i = 0; i < scene->size(); i++) {
    scene_new->at(i).x = scene->at(i).x;
    scene_new->at(i).y = scene->at(i).y;
    scene_new->at(i).z = scene->at(i).z;
}
   
  pcl::ExtractIndices<pcl::PointXYZ> extractn;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);
  pcl::PCDWriter writer;
  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::PointCloud <pcl::Normal>::Ptr norm_c (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (scene_new);
  normal_estimator.setKSearch (100);
  normal_estimator.compute (*normals);
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (100);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (100);
  reg.setInputCloud (scene_new);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (7.0 / 180.0 * M_PI);//5
  reg.setCurvatureThreshold (7.0);
 
  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);
  int max_area=0;int max_ind;
  pcl::PointCloud<pcl::PointXYZ>::Ptr filt (new pcl::PointCloud<pcl::PointXYZ> ());
  float curv[clusters.size()];

 for(int j = 0;j<clusters.size();j++)
  {
//EXtract cloud plane for the given cluster
  pcl::PointCloud <pcl::Normal>::Ptr norm_c (new pcl::PointCloud <pcl::Normal>);
 pcl::IndicesPtr indices_ptr (new std::vector<int> (clusters[j].indices.size ())); 
  for (int i = 0; i < indices_ptr->size (); i++) 
          (*indices_ptr)[i] = clusters[j].indices[i]; 

  extractn.setInputCloud (scene_new);
  extractn.setIndices (indices_ptr);
  extractn.setNegative (false);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  extractn.filter (*cloud_plane);

//calculate normals for the given cluster
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm;
  norm.setSearchMethod (tree);
  norm.setInputCloud (cloud_plane);
  norm.setKSearch (100);
  norm.compute (*norm_c);
   // Concatenate the XYZ and normal fields*
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*cloud_plane, *norm_c, *cloud_with_normals);
 
// REmove NaNs from POintcloud
  std::vector<int> indices;
  removeNaNNormalsFromPointCloud (*norm_c, *norm_c, indices);

        float norm_x = 0;
        for(int k=0;k<norm_c->size();k++)
	{
          norm_x = norm_x + norm_c->points[k].curvature;
	}

   curv[j] = norm_x/norm_c->size();

 // std::cout<<"norm "<<curv[j]<<endl;

}

// Extract all useless clouds from full cloud
int size_curve=0;
for(int t=0;t<clusters.size();t++)
{
if(curv[t]<=0.08)
   size_curve = size_curve + clusters[t].indices.size ();
}


pcl::IndicesPtr indices_ptr (new std::vector<int> (size_curve)); 
int count = 0;
for(int t=0;t<clusters.size();t++)
{
if(curv[t]<=0.04)
{
  for (int i = 0; i < clusters[t].indices.size (); i++) {
          (*indices_ptr)[count] = clusters[t].indices[i];count++;} 
}
  
}
  extractn.setInputCloud (scene_new);
  extractn.setIndices (indices_ptr);
  extractn.setNegative (false);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  extractn.filter (*filt);


// Write segmented scene to file

  //pcl::visualization::CloudViewer viewer ("Cluster viewer");
  //viewer.showCloud(filt);
  //while (!viewer.wasStopped ())
  //{
  //}


scene_crop->resize(filt->size());
for (size_t i = 0; i < filt->size(); i++) {
    scene_crop->at(i).x = filt->at(i).x;
    scene_crop->at(i).y = filt->at(i).y;
    scene_crop->at(i).z = filt->at(i).z;
}

////////////////////////////////////////////////////////////////Resume Object DEtection Code//////////////////////////////////////////////////////////
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;  pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
  norm_est.setKSearch (50);
  norm_est.setSearchMethod(kdtree);
  norm_est.setInputCloud (scene_crop);
  norm_est.compute (*scene_normals);

  //filter voxel
  pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
  sor.setInputCloud (scene_crop);
  sor.setLeafSize (0.03f,0.03f,0.03f);
  sor.filter (*scene_keypoints);
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est1; 
  norm_est1.setRadiusSearch (0.15);
  norm_est1.setSearchMethod(kdtree);
  norm_est1.setInputCloud (scene_keypoints);
  norm_est1.compute (*scenefilt_normals);
  
  pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud (scene_keypoints);
  fpfh.setInputNormals (scenefilt_normals);
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr treeFPFH_scene (new pcl::search::KdTree<pcl::PointXYZRGBA>);
  fpfh.setSearchMethod (treeFPFH_scene);
  fpfh.setRadiusSearch (0.15);
  fpfh.compute (*scene_descriptors);
  
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_descriptors->size () << std::endl;
  //fpfhs->points.size () should have the same size as the input cloud->points.size ()*
  
int cluster_grp;
ros::NodeHandle sh;  ros::Publisher pub = sh.advertise<geometry_msgs::Vector3> ("ModelPos", 1);
ros::Rate loop_rate(10);
  for (int l = 0; l < 42; ++l)
{
  for (int k=0;k<2;k++)
{
  //Load CAD clouds
if(k==0)
  st1 = "/home/nus/catkin_ws/iros_data/chord";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd
  else if (k==1)
   st1 = "/home/nus/catkin_ws/iros_data/stub";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd

  stringstream sm;
int h;
if (l==0)
    h = 14;//3; //cluster 1
else if (l==1)
    h = 5;//cluster 2
else if (l==2)
    h = 8;//cluster 3
else if (l==3)
    h = 11;//cluster 3
else if (l==4)
    h = 12;//cluster 3
else if (l==5)
    h = 14;//cluster 3
else if (l==6)
    h = 16;//cluster 3
else if (l==7)
    h = 32;//cluster 3
else if (l==8)
    h = 33;//cluster 3
  sm << l;
  filename = st1 + sm.str() + st2;
  

  if (pcl::io::loadPCDFile (filename, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    //showHelp (argv[0]);
    //return (-1);
  }
   st1 = "/home/nus/catkin_ws/iros_data/";///home/nus/catkin_ws/stub_poses/3.pcd    only_chord.pcd

  filename = st1 + sm.str() + st2;
  pcl::io::loadPCDFile (filename, *full_pose);
  
// REmove NaNs from POintcloud

  pcl::removeNaNFromPointCloud(*model,*model, indices); 
  pcl::removeNaNFromPointCloud(*full_pose,*full_pose, indices); 
 //  Compute Normals
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est_model;
  norm_est_model.setInputCloud (model);
  norm_est_model.setRadiusSearch (0.15);
  pcl::search::KdTree<PointType>::Ptr kdtree_model(new pcl::search::KdTree<PointType>);
  norm_est_model.setSearchMethod(kdtree_model);
  norm_est_model.compute (*model_normals);
  //  Downsample Clouds to Extract keypoints
  sor.setInputCloud (model);
  sor.setLeafSize (0.02,0.02,0.02);
  sor.filter (*model_keypoints);
  //  Compute Normals
  norm_est_model.setInputCloud (model_keypoints);
  norm_est_model.setRadiusSearch (0.15);
  norm_est_model.setSearchMethod(kdtree_model);
  norm_est_model.compute (*modelfilt_normals);
 // Add another keypoint extractor
 pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
  //  Compute Descriptor for keypoints
  
  pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud (model_keypoints);
  fpfh.setInputNormals (modelfilt_normals);
  fpfh.setSearchMethod(tree);
  fpfh.setRadiusSearch (0.15);
  fpfh.compute (*model_descriptors);

  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_descriptors->size () << std::endl;
  //cout<<model_descriptors->at(0)<<endl;
  //  Find Model-Scene Correspondences with KdTree
  //
  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
  pcl::KdTreeFLANN<DescriptorType1> match_search;
  match_search.setInputCloud (model_descriptors);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {    
    std::vector<int> neigh_indices (2);
    std::vector<float> neigh_sqr_dists (2);
    if (!pcl_isfinite (scene_descriptors->points[i].histogram[0])) //skipping NaNs
    {
      continue;
    }

    int found_neighs = match_search.nearestKSearch (scene_descriptors->points[i], 2, neigh_indices, neigh_sqr_dists);
    double tau = neigh_sqr_dists[0]/neigh_sqr_dists[1];
    if(found_neighs >= 1 && tau<=1) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
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
    rf_est.setSearchSurface (scene_crop);
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

 double prev_score; double prev_score2;double score2;double score;Eigen::Matrix4f Final_pose;
  std::cout << "Model instances found: " << rototranslations.size () << std::endl;
    CORRS = CORRS+rototranslations.size ();
 // cout<<"corrs "<<CORRS<<endl;
  if(rototranslations.size () >0)
  {
  for (size_t i = 0; i < rototranslations.size (); ++i)
  {
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
  icp.setInputTarget (scene_crop);
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
    icp.setInputTarget (scene_crop);
    icp.align (*rotated_full);
    }
    std::stringstream ss_cloud;
    //ss_cloud << "instance" << i;
    
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
    icp.setInputTarget (scene_crop);
    icp.align (*rotated_full);

    score2 = icp.getFitnessScore(); 
    if(counter2==1)
    {
	prev_score2  = score2;
    }
     counter2++;
    if ((score2 < prev_score2 || score2==prev_score2) && counter2>1 && score2<0.006) //
    {
     prev_score2 = score2;
     icp_flag = false;
     printf ("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
     cout<<"Pose "<< l<< " Matched to Scene "<<m<<endl;
    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Full_pose_new (0,0), Full_pose_new(0,1), Full_pose_new (0,2),Full_pose_new(0,3));
    printf ("        R = | %6.3f %6.3f %6.3f %6.3f | \n", Full_pose_new (1,0), Full_pose_new(1,1), Full_pose_new(1,2),Full_pose_new(1,3));
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Full_pose_new (2,0), Full_pose_new (2,1), Full_pose_new (2,2),Full_pose_new(2,3));
    printf ("            | %6.3f %6.3f %6.3f %6.3f | \n", Full_pose_new (3,0), Full_pose_new (3,1), Full_pose_new (3,2),Full_pose_new(3,3));
    printf ("\n");
     } 
   if (icp_flag==false)
    {
 COUNTER++;
  //cout<<"No of matches found "<< COUNTER<<"POSE NO "<< l<<endl;


 //////////////////////////////////////////////////////////////////////////////////// Show bounding box/////////////////////////////////////////////////////////
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
  normal_estimator.setKSearch (30);
  normal_estimator.compute (*normals);
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (500);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (bounding_box);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (5.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (5.0);

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
  

  //Compute Centroid
  
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
    //pcl::transformPointCloud (*rotated_full, *translate_x, move_to_origin);
    //pcl::transformPointCloud (*translate_x, *full_constraint, x_constraint);
    //pcl::transformPointCloud (*full_constraint, *translate_y, y_constraint);
    move_to_origin(0,3) = centroidnew[0];
    move_to_origin(1,3) = centroidnew[1];
    move_to_origin(2,3) = centroidnew[2];
   // pcl::transformPointCloud (*full_constraint,*translate_y, move_to_origin);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 //  Visualization
  string input;
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
    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_full_color_handler (rotated_full, 50, 155, 200);
    viewer.addPointCloud (rotated_full, rotated_full_color_handler, "rotated view keypoints");
     
 // draw the cloud and the box
    //viewer.addPointCloud(point_cloud_ptr);
   // viewer.addCube(tfinal, qfinal, max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z,"cube", 0);
  while (!viewer.wasStopped ())
  {
    viewer.spin();

    
  }

	  //get input from user
	  cout<<"Enter y or n "<<endl;
	  cin>>input;
	  if(input=="y")
	  {
	  cout<<"yes"<<endl;
	  
	  //publish centroid data
	  
	  while(ros::ok())
	  {
	  geometry_msgs::Vector3 center;
	  center.x = 1+centroid[0];
	  center.y = centroid[1];
	  center.z = -0.8+centroid[2];
	  pub.publish(center);
	  //ros::spinOnce();
	  loop_rate.sleep();
	  
	  }
	  }
		if(input=="n")
	  {
	  cout<<"no"<<endl;
	  icp_flag = true;
	  }
	 

  

    } //if score previous
    } 
    }
    }//if rototranslantation
    }
   }

   }//End of loop for a scene
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
     //cloud_cb();
     // Spin
     ros::spin(); 
   }
