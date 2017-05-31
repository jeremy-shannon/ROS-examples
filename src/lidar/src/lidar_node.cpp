// ROS Point Cloud DEM Generation
// MacCallister Higgins

#include <cmath>
#include <vector>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <sstream>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#define IMAGE_HEIGHT	701
#define IMAGE_WIDTH	801
#define BIN		0.100

// choices for min/max point height (z) to consider
#define MIN_Z -2.0    // previous -1.9
#define MAX_Z  0.5    // previous 0.7

// capture car dimensions (for removing it from point cloud)
#define CAPTURE_CAR_FRONT_X 2.0
#define CAPTURE_CAR_REAR_X -1.5
#define CAPTURE_CAR_LEFT_Y 1.0
#define CAPTURE_CAR_RIGHT_Y -1.0

using namespace cv;

// Global Publishers/Subscribers
ros::Subscriber subPointCloud;
//ros::Publisher pubPointCloud;
ros::Subscriber subObjRTK;
ros::Subscriber subCapFRTK;
ros::Subscriber subCapRRTK;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_grid (new pcl::PointCloud<pcl::PointXYZ>);
//sensor_msgs::PointCloud2 output;
nav_msgs::Odometry objRTK;
nav_msgs::Odometry capFRTK;
nav_msgs::Odometry capRRTK;

double heightArray[IMAGE_HEIGHT][IMAGE_WIDTH];

cv::Mat *heightmap, *cluster_img;
std::vector<int> compression_params;

int fnameCounter;
double lowest;

// odometry
double obj_gps_x, obj_gps_y, cap_gps_front_x, cap_gps_front_y, cap_gps_rear_x, cap_gps_rear_y;

// map meters to 0->255
int map_m2i(double val){
  return (int)round((val - MIN_Z + 2.5)/(MAX_Z - MIN_Z + 2.5) * 255);
}

// map meters to index
// returns 0 if not in range, 1 if in range and row/column are set
int map_pc2rc(double x, double y, int *row, int *column){
  // Find x -> row mapping
  *row = (int)round(floor(((((IMAGE_HEIGHT*BIN)/2.0) - x)/(IMAGE_HEIGHT*BIN)) * IMAGE_HEIGHT));	
  // obviously can be simplified, but leaving for debug
  // Find y -> column mapping
  *column = (int)round(floor(((((IMAGE_WIDTH*BIN)/2.0) - y)/(IMAGE_WIDTH*BIN)) * IMAGE_WIDTH));
  // Return success
  return 1;
}

// map index to meters
// returns 0 if not in range, 1 if in range and x/y are set
int map_rc2pc(double *x, double *y, int row, int column){
  // Check if falls within range
  if(row >= 0 && row < IMAGE_HEIGHT && column >= 0 && column < IMAGE_WIDTH){
    // Find row -> x mapping
    *x = (double)(BIN*-1.0 * (row - (IMAGE_HEIGHT/2.0)));	// this one is simplified
    // column -> y mapping
    *y = (double)(BIN*-1.0 * (column - (IMAGE_WIDTH/2.0)));
    // Return success
    return 1;
  }
  return 0;
}


// main generation function
void DEM(const sensor_msgs::PointCloud2ConstPtr& pointCloudMsg)
{
  ROS_DEBUG("Point Cloud Received");
  // ROS_INFO("Position-> obj_lidar_x: [%f], obj_lidar_y: [%f], cf_x: [%f], cf_y: [%f], cr_x: [%f], cr_y: [%f],",
  //   obj_gps_x, obj_gps_y, cap_gps_front_x, cap_gps_front_y, cap_gps_rear_x, cap_gps_rear_y);

  // Calculate location of object car in relation to capture car
  double theta = atan2((cap_gps_front_y-cap_gps_rear_y),(cap_gps_front_x-cap_gps_rear_x));
  double d_y = obj_gps_y - cap_gps_front_y;
  double d_x = obj_gps_x - cap_gps_front_x;
  double obj_lidar_y = d_x*sin(-theta) + d_y*cos(-theta);
  double obj_lidar_x = d_x*cos(-theta) - d_y*sin(-theta) -1.0; // lidar/gps difference

  // Convert from ROS message to PCL point cloud
  pcl::fromROSMsg(*pointCloudMsg, *cloud);

  ///////////////////////////////// DOWNSAMPLE ////////////////////////////////////////////
  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.1f, 0.1f, 0.1f);
  vg.filter (*cloud_filtered);
  //ROS_INFO_STREAM("PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points.");

  ///////////////////// GROUND PLANE SEGMENTATION AND REMOVAL ///////////////////////////
  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.5);

  int nr_points = (int) cloud_filtered->points.size ();
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      ROS_INFO_STREAM("Could not estimate a planar model for the given dataset.");
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    //ROS_INFO_STREAM("PointCloud representing the planar component: " << cloud_plane->points.size () << " data points.");

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  ///////////////////////////////// LIMIT ////////////////////////////////////////////
  // remove points from the cloud below MIN_Z and above MAX_Z
  //ROS_INFO_STREAM("Points before limit: " << cloud_filtered->points.size ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_limited1 (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond1 (new
    pcl::ConditionAnd<pcl::PointXYZ> ());
  range_cond1->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, MIN_Z)));
  range_cond1->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, MAX_Z)));
  // build the filter
  pcl::ConditionalRemoval<pcl::PointXYZ> condrem1 (range_cond1);
  condrem1.setInputCloud (cloud_filtered);
  //condrem.setKeepOrganized(true);
  // apply filter
  condrem1.filter (*cloud_limited1);
  //ROS_INFO_STREAM("Points after limit 1: " << cloud_limited1->points.size ());

  // remove points corresponding to the capture vehicle
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_limited (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConditionOr<pcl::PointXYZ>::Ptr range_cond (new
    pcl::ConditionOr<pcl::PointXYZ> ());
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::GT, CAPTURE_CAR_FRONT_X)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::LT, CAPTURE_CAR_REAR_X)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::GT, CAPTURE_CAR_LEFT_Y)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::LT, CAPTURE_CAR_RIGHT_Y)));
  // build the filter
  pcl::ConditionalRemoval<pcl::PointXYZ> condrem (range_cond);
  condrem.setInputCloud (cloud_limited1);
  //condrem.setKeepOrganized(true);
  // apply filter
  condrem.filter (*cloud_limited);
  //ROS_INFO_STREAM("Points after limit: " << cloud_limited->points.size ());

  ///////////////////////////////// CLUSTER ////////////////////////////////////////////
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_limited);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (1.75); // meters
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (2500);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_limited);
  ec.extract (cluster_indices);

  ROS_INFO_STREAM(cluster_indices.size() << " clusters extracted");

  // clear heightArray
  lowest = FLT_MAX;
  for(int i = 0; i < IMAGE_HEIGHT; ++i) {
    for(int j = 0; j < IMAGE_WIDTH; ++j) {
      heightArray[i][j] = (double)(-FLT_MAX);
    }
  }

  // clear the heightmap image
  for(int i = 0; i < IMAGE_HEIGHT; ++i) {
    for(int j = 0; j < IMAGE_WIDTH; ++j) {
      cv::Vec3b &pixel = heightmap->at<cv::Vec3b>(i,j);
      pixel[0] = 0;
      pixel[1] = 0;
      pixel[2] = 0;
    }
  }

  int cluster_index = 0;
  // iterate through clusters
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    int row, column;
    Eigen::Vector4f centroid;

    //ROS_INFO_STREAM("cluster size: " << it->indices.size() );

    // compute centroid of cluster
    pcl::compute3DCentroid(*cloud_limited, it->indices, centroid);
    //ROS_INFO_STREAM("centroid: " << centroid[0] << "," << centroid[1] << "," << centroid[2]);
    
    // determine whether cluster is object car based on distance from 
    // cluster centroid to object car coordinates (in lidar frame)
    int circle_radius, circle_thickness;
    circle_radius = 4;
    circle_thickness = 1;
    const char* file_path = "images/noncar/";
    double dx = centroid[0] - obj_lidar_x;
    double dy = centroid[1] - obj_lidar_y;
    double dist = sqrt(pow(dx,2) + pow(dy,2));
    if (dist < 2.5) {
      // object car found
      circle_radius = 6;
      circle_thickness = 2;
      file_path = "images/car/";    
    }


    // iterate through points in the cluster
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
      // If the point is within the image size bounds
      if(map_pc2rc(cloud_limited->points[*pit].x, cloud_limited->points[*pit].y, &row, &column) == 1 &&
                  row >= 0 && 
                  row < IMAGE_HEIGHT && 
                  column >=0 && 
                  column < IMAGE_WIDTH) {
        // if the height of the current point is the highest at this index...
        if(cloud_limited->points[*pit].z > heightArray[row][column]) {
          // add the height of the point to the height array
          heightArray[row][column] = cloud_limited->points[*pit].z;
        }
      }
    }

    int num_cluster_image_pixels = 0;

    // clear the cluster image
    for(int i = 0; i < 64; ++i) {
      for(int j = 0; j < 64; ++j) {
        uchar &pixel = cluster_img->at<uchar>(i,j);
        pixel = 0;
      }
    }
    // add heightArray values to cluster image
    for(int i = 0; i < 64; ++i) {
      for(int j = 0; j < 64; ++j) {
        // calculate heightArray offset - 64x64 square centered at 
        // cluster centroid (converted to heightMap coords)
        int centroid_row, centroid_col;
        map_pc2rc(centroid[0], centroid[1], &centroid_row, &centroid_col);
        int offset_i = centroid_row - 32 + i;
        int offset_j = centroid_col - 32 + j;
        if (offset_i >= 0 && offset_i < IMAGE_HEIGHT && 
            offset_j >= 0 && offset_j < IMAGE_WIDTH) {
          // Add point to image
          uchar &pixel = cluster_img->at<uchar>(i,j);
          if(heightArray[offset_i][offset_j] > -FLT_MAX) {
            pixel = map_m2i(heightArray[offset_i][offset_j]);
            num_cluster_image_pixels++;
          }
        }
      }
    }
    // add heightArray values to heightmap image
    for(int i = 0; i < IMAGE_HEIGHT; ++i) {
      for(int j = 0; j < IMAGE_WIDTH; ++j) {
        // Add point to image
        cv::Vec3b &pixel = heightmap->at<cv::Vec3b>(i,j);
        if(heightArray[i][j] > -FLT_MAX) {
          // modulo is to get different colors for different clusters
          pixel[cluster_index%3] = map_m2i(heightArray[i][j]);
        }
        // clear heightArray at this location, to prepare for next cluster
        heightArray[i][j] = -FLT_MAX;
      }
    }

    if (num_cluster_image_pixels > 9) {
      // Save cluster image to disk
      char filename[100];
      snprintf(filename, 100, "%simage_%d-%d.png", file_path, fnameCounter, cluster_index);
      cv::imwrite(filename, *cluster_img, compression_params);
    }

    // Draw a pretty little circle around the cluster centroid (big and bold if ID'd as obj car)
    int centroid_row, centroid_col; 
    map_pc2rc(centroid[0], centroid[1], &centroid_row, &centroid_col); 
    cv::circle(*heightmap, Point(centroid_col, centroid_row), circle_radius, 
               Scalar(255,255,255), circle_thickness);
    
    cluster_index++;
  }

  // Draw a pretty little circle around the lidar
  int lidar_origin_x, lidar_origin_y;
  map_pc2rc(0.0, 0.0, &lidar_origin_y, &lidar_origin_x); 
  cv::circle(*heightmap, Point(lidar_origin_x,lidar_origin_y), 4, Scalar(255,255,0), 1);

  // Draw a pretty box around the capture car position
  int cap_front_rc, cap_rear_rc, cap_left_rc, cap_right_rc;
  map_pc2rc(CAPTURE_CAR_FRONT_X, CAPTURE_CAR_LEFT_Y, &cap_front_rc, &cap_left_rc);
  map_pc2rc(CAPTURE_CAR_REAR_X, CAPTURE_CAR_RIGHT_Y, &cap_rear_rc, &cap_right_rc);
  cv::rectangle(*heightmap, Point(cap_left_rc, cap_front_rc), Point(cap_right_rc, cap_rear_rc), Scalar(255,255,0), 1);

  // Draw a pretty little circle on the object car
  int obj_lidar_x_rc, obj_lidar_y_rc;
  if (map_pc2rc(obj_lidar_x, obj_lidar_y, &obj_lidar_y_rc, &obj_lidar_x_rc) == 1 &&
                 obj_lidar_y_rc >= 0 && 
                 obj_lidar_y_rc < IMAGE_HEIGHT && 
                 obj_lidar_x_rc >=0 && 
                 obj_lidar_x_rc < IMAGE_WIDTH) {
    cv::circle(*heightmap, Point(obj_lidar_x_rc,obj_lidar_y_rc), 4, Scalar(255,0,255), 1);
  }

  //ROS_INFO_STREAM("gps: " << obj_lidar_x << "," << obj_lidar_y << "   pix: " << obj_lidar_x_rc << "," << obj_lidar_y_rc);
  ++fnameCounter;

  // Display image
  //cv::imshow("Height Map", *heightmap);

  // Output height map to point cloud for python node to parse to PNG
  /*
  pcl::toROSMsg(*cloud_grid, output);
  output.header.stamp = ros::Time::now();
  output.header.frame_id = "velodyne";
  pubPointCloud.publish(output);
  */

}

// received RTK message from object vehicle
void ObjRTKRecd(const nav_msgs::Odometry::ConstPtr& objRTKmsg) {
  obj_gps_x = objRTKmsg->pose.pose.position.x;
  obj_gps_y = objRTKmsg->pose.pose.position.y;
  // ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", 
  //   objRTKmsg->pose.pose.position.x,
  //   objRTKmsg->pose.pose.position.y, 
  //   objRTKmsg->pose.pose.position.z);
}
// received RTK message from front of capture vehicle
void CapFRTKRecd(const nav_msgs::Odometry::ConstPtr& capFRTKmsg) {
  cap_gps_front_x = capFRTKmsg->pose.pose.position.x;
  cap_gps_front_y = capFRTKmsg->pose.pose.position.y;
}
// received RTK message from rear of capture vehicle
void CapRRTKRecd(const nav_msgs::Odometry::ConstPtr& capRRTKmsg) {
  cap_gps_rear_x = capRRTKmsg->pose.pose.position.x;
  cap_gps_rear_y = capRRTKmsg->pose.pose.position.y;
}

int main(int argc, char** argv)
{
  ROS_INFO("Starting LIDAR Node");
  ros::init(argc, argv, "lidar_node");
  ros::NodeHandle nh;

  // Setup output cloud
  /*
  cloud_grid->width  = IMAGE_WIDTH;
  cloud_grid->height = IMAGE_HEIGHT;
  cloud_grid->points.resize (cloud_grid->width * cloud_grid->height);
  */

  // Setup images
  cv::Mat map(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
  heightmap = &map;
  cv::Mat map2(64, 64, CV_8UC1, cv::Scalar(0, 0, 0));
  cluster_img = &map2;
  cvNamedWindow("Height Map", CV_WINDOW_AUTOSIZE);
  cvStartWindowThread();
  cv::imshow("Height Map", *heightmap);

  // Setup Image Output Parameters
  fnameCounter = 0;
  lowest = FLT_MAX;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  // Setup indicies in point clouds
/*
  int index = 0;
  ROS_INFO("x");
  double x, y;
  for(int i = 0; i < IMAGE_HEIGHT; ++i){
    for(int j = 0; j < IMAGE_WIDTH; ++j){
      index = i * j;
      (void)map_rc2pc(&x, &y, i, j);
      cloud_grid->points[index].x = x;
      cloud_grid->points[index].y = y;
      cloud_grid->points[index].z = (-FLT_MAX);
      // Temp storage
      heightArray[i][j] = (-FLT_MAX);
      }
    }
*/

  subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 2, DEM);
  //pubPointCloud = nh.advertise<sensor_msgs::PointCloud2> ("/heightmap/pointcloud", 1);
  subObjRTK = nh.subscribe<nav_msgs::Odometry>("/objects/obs1/rear/gps/rtkfix", 2, ObjRTKRecd);
  subCapFRTK = nh.subscribe<nav_msgs::Odometry>("/objects/capture_vehicle/front/gps/rtkfix", 2, CapFRTKRecd);
  subCapRRTK = nh.subscribe<nav_msgs::Odometry>("/objects/capture_vehicle/rear/gps/rtkfix", 2, CapRRTKRecd);

  ros::spin();

  return 0;
}
