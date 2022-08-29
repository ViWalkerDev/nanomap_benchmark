
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/storage_options.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>


#include <octomap/OcTree.h>
#include <octomap_ros/conversions.hpp>

#include <openvdb/tools/TopologyToLevelSet.h>
#include <nanomap/manager/Manager.h>
#include <nanomap/map/OccupancyMap.h>
#include <nanomap/nanomap.h>
#include <nanomap/sensor/SensorData.h>
#include <nanomap/sensor/FrustumData.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Ray.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/math/DDA.h>
#include <nanovdb/util/IO.h>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <string>
#include <fstream>

using rosbag2_cpp::converter_interfaces::SerializationFormatConverter;
using ValueT = float;
using Pose = nanomap::Pose;
using FloatGrid = openvdb::FloatGrid;

using TreeT = openvdb::FloatGrid::TreeType;
using IterType = TreeT::ValueOnIter;

using SensorData = nanomap::sensor::SensorData;
using FrustumData = nanomap::sensor::FrustumData;
using Map = nanomap::map::Map;
using RayT  = openvdb::math::Ray<double>;
using Vec3T = RayT::Vec3Type;
using DDAT  = openvdb::math::DDA<RayT, 0>;
nanomap::Pose tf2pose(geometry_msgs::msg::TransformStamped tf){
    nanomap::Pose pose;
    pose.position = Eigen::Vector3f(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z );
    pose.orientation = Eigen::Quaternionf(tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w );

    return pose;
}

void loadVDBConfig(std::string vdbConfigStr, std::string sensorConfigStr,
              float& _mappingRes, float& _oddsMiss, float& _oddsHit,
              float& _oddsThresMin, float& _oddsThresMax, float& _maxRange){
                std::cout <<"reading in vdb config file: " << vdbConfigStr << std::endl;
                std::ifstream *inputConfig = new std::ifstream(vdbConfigStr.c_str(), std::ios::in | std::ios::binary);
                float mappingRes, oddsThresMin, oddsThresMax;
                std::string line;
                *inputConfig >> line;
                if(line.compare("#config") != 0){
                  std::cout << "Error: first line reads [" << line << "] instead of [#config]" << std::endl;
                  delete inputConfig;
                  return;
                }
                while(inputConfig->good()) {
                    *inputConfig >> line;
                    if (line.compare("MappingRes:") == 0){
                      *inputConfig >> mappingRes;
                      _mappingRes = mappingRes;
                    }else if (line.compare("ProbHitThres:") == 0){
                      *inputConfig >> oddsThresMax;
                      _oddsThresMax = oddsThresMax;
                    }else if (line.compare("ProbMissThres:") == 0){
                      *inputConfig >> oddsThresMin;
                      _oddsThresMin = oddsThresMin;
                    }else if (line.compare("#endconfig")==0){
                      break;
                    }
                  }
                inputConfig->close();
                delete inputConfig;
                std::cout <<"reading in sensor config file: " << sensorConfigStr << std::endl;
                std::ifstream *inputSensor = new std::ifstream(sensorConfigStr.c_str(), std::ios::in | std::ios::binary);
                float oddsMiss, oddsHit, maxRange;
                *inputSensor >> line;
                while(inputSensor->good()) {
                    *inputSensor >> line;
                    if (line.compare("MaxRange:") == 0){
                      *inputSensor >> maxRange;
                      _maxRange = maxRange;
                    }else if (line.compare("ProbHit:") == 0){
                      *inputSensor >> oddsHit;
                      _oddsHit = oddsHit;
                    }else if (line.compare("ProbMiss:") == 0){
                      *inputSensor >> oddsMiss;
                      _oddsMiss = oddsMiss;
                    }else if (line.compare("#endconfig")==0){
                      break;
                    }
                  }
                inputSensor->close();
                delete inputSensor;
}

void octomapBuildMapBulkScan(Eigen::Matrix3f frameTransform,
        std::vector<std::shared_ptr<sensor_msgs::msg::PointCloud2>> &clouds,
        std::vector<geometry_msgs::msg::PoseStamped> &poses,
        float maxRange,
        octomap::OcTree *octree,
        bool discretize) {


        Eigen::Quaternionf frameQuat(frameTransform);
    for (int i = 0; i < clouds.size(); i++) {
        octomap::Pointcloud ompc;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*(clouds[i]), "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*(clouds[i]), "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*(clouds[i]), "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z){
          if (!std::isnan (*iter_x) && !std::isnan (*iter_y) && !std::isnan (*iter_z)){

            ompc.push_back(*iter_x, *iter_y, *iter_z);
          }
        }
        octomap::point3d sensorOrigin;
        sensorOrigin.x() = 0.0;
        sensorOrigin.y() = 0.0;
        sensorOrigin.z() = 0.0;
        Eigen::Quaternionf poseQuat(poses[i].pose.orientation.w,
                                                 poses[i].pose.orientation.x,
                                                 poses[i].pose.orientation.y,
                                                 poses[i].pose.orientation.z
                                                );
        Eigen::Quaternionf sensorQuat = poseQuat.normalized()*frameQuat.inverse();
        octomap::pose6d sensorPose(octomath::Vector3(poses[i].pose.position.x,
                                                 poses[i].pose.position.y,
                                                 poses[i].pose.position.z
                                               ),
                                   octomath::Quaternion(sensorQuat.w(),
                                              sensorQuat.x(),
                                              sensorQuat.y(),
                                              sensorQuat.z()
                                             ));

        octree->insertPointCloud(ompc, sensorOrigin, sensorPose, maxRange, false, discretize);
      if(i>= clouds.size()-1){
	break;
	}
    }
}


void populateTempGrid(openvdb::FloatGrid::Ptr grid, openvdb::FloatGrid::Ptr tempGrid, openvdb::FloatGrid::Accessor& tempAcc,
			int cloudWidth, int cloudHeight, int cloudPointStep, unsigned char* cloudPtr,
                                        const nanomap::Pose& pose,
					float max_range,
					float logodds_miss,
					float logodds_hit,
					float gridRes,
					Eigen::Matrix<float, 3, 3> frameTransform){
  RayT ray;
  DDAT dda;

  openvdb::Vec3d ray_origin_world(pose.position(0), pose.position(1), pose.position(2));

  const Vec3T ray_origin_index(grid->worldToIndex(ray_origin_world));

  openvdb::Vec3d ray_direction;
  bool max_range_ray;
  openvdb::Vec3d x;
  double ray_length;
  float max_time = max_range/gridRes;
  float pointx, pointy, pointz;
  Eigen::Matrix<float,3,3> rotation = pose.orientation.normalized().toRotationMatrix()*frameTransform;

  // Probability update lambda for empty grid elements
  auto miss = [&prob_miss = logodds_miss](float& voxel_value, bool& active) {
    voxel_value += prob_miss;
    active = true;
  };

  // Probability update lambda for occupied grid elements
  auto hit = [&prob_hit = logodds_hit](float& voxel_value, bool& active) {
    voxel_value += prob_hit;
    active = true;
  };
  // Raycasting of every point in the input cloud
  for (int i = 0; i < cloudWidth*cloudHeight; i++)
  {
    unsigned char* byte_ptr = cloudPtr + i*cloudPointStep;
    pointx = *(reinterpret_cast<float*>(byte_ptr+0));
    pointy = *(reinterpret_cast<float*>(byte_ptr+4));
    pointz = *(reinterpret_cast<float*>(byte_ptr+8));

    max_range_ray = false;
    ray_direction = openvdb::Vec3d(
		  rotation(0,0)*pointx+rotation(0,1)*pointy+rotation(0,2)*pointz,
                  rotation(1,0)*pointx+rotation(1,1)*pointy+rotation(1,2)*pointz,
                  rotation(2,0)*pointx+rotation(2,1)*pointy+rotation(2,2)*pointz);

    ray_length = ray_direction.length()/gridRes;
    if(ray_length  < max_time){
      ray_direction.normalize();
      ray.setEye(ray_origin_index);
      ray.setDir(ray_direction);
      dda.init(ray,0,ray_length);
      openvdb::Coord voxel = dda.voxel();
      while(dda.step()){
  	     tempAcc.modifyValueAndActiveState(voxel, miss);
         voxel = dda.voxel();
      }
      if(dda.time()<max_time){
  	     tempAcc.modifyValueAndActiveState(voxel, hit);
      }
    }
  }
}

void integrateTempGrid(openvdb::FloatGrid::Ptr grid, openvdb::FloatGrid::Ptr tempGrid, openvdb::FloatGrid::Accessor& acc,
			float emptyClampThres, float occClampThres, float logodds_thres_min, float logodds_thres_max){
  float tempValue;
  // Probability update lambda for occupied grid elements
  auto update = [&prob_thres_max = logodds_thres_max, &prob_thres_min = logodds_thres_min,
		             &occ_clamp = occClampThres, &empty_clamp = emptyClampThres, &temp_value = tempValue]
                 (float& voxel_value, bool& active) {
    voxel_value += temp_value;
    if (voxel_value > occ_clamp)
    {
      voxel_value = occ_clamp;
    }else if(voxel_value < empty_clamp){
      voxel_value = empty_clamp;
    }

    if(voxel_value > prob_thres_max){
      active = true;
    }else if(voxel_value < prob_thres_min){
      active = false;
    }
  };
  // Integrating the data of the temporary grid into the map using the probability update functions
  for (openvdb::FloatGrid::ValueOnCIter iter = tempGrid->cbeginValueOn(); iter; ++iter)
  {
    tempValue = iter.getValue();
    if (tempValue!=0.0)
    {
      acc.modifyValueAndActiveState(iter.getCoord(), update);
    }
  }
return;
}

void vdbMappingTest(std::vector<std::shared_ptr<sensor_msgs::msg::PointCloud2>> &clouds,
        	                   std::vector<geometry_msgs::msg::PoseStamped>                 &poses,
	 	                         openvdb::FloatGrid::Ptr                               grid,
		                         openvdb::FloatGrid::Accessor                       gridAcc,
		                         float                                             _gridRes,
                             float                                            _maxRange,
                             float                                             oddsMiss,
                             float                                              oddsHit,
                             float                                         oddsThresMin,
                             float                                         oddsThresMax,
                             float                                        occClampThres,
                             float                                      emptyClampThres,
		                         float&                                       mapUpdateTime,
		                         float&                                    cloudProcessTime,
			                       Eigen::Matrix<float, 3, 3>                  frameTransform,
                             int                                                  start,
                             int                                                    end){

   float gridRes = _gridRes;
   float logOddsMiss = log(oddsMiss)-log(1-oddsMiss);
   float logOddsHit = log(oddsHit)-log(1-oddsHit);
   float logOddsThresMin = log(oddsThresMin) - log(1-oddsThresMin);
   float logOddsThresMax = log(oddsThresMax) - log(1-oddsThresMax);
   float logOccClampThres = log(occClampThres)-log(1-occClampThres);
   float logEmptyClampThres = (log(emptyClampThres)-log(1-emptyClampThres));
   float maxRange =  _maxRange;
   std::chrono::duration<double, std::milli> delay;
   //define timers;
   int index;
   int indexTarget =start;
   int indexEnd = end;
   nanomap::Pose pose;
   geometry_msgs::msg::TransformStamped sensorToWorldTf;
   Eigen::Quaterniond quat;
   Eigen::Vector3d pos;
   auto handle_start = std::chrono::high_resolution_clock::now();
   auto handle_end = std::chrono::high_resolution_clock::now();
   for(auto itr = clouds.begin(); itr != clouds.end(); itr++){
     index = std::distance(clouds.begin(), itr);
       openvdb::FloatGrid::Ptr tempGrid = openvdb::FloatGrid::create(0.0);
       openvdb::FloatGrid::Accessor tempAcc = tempGrid->getAccessor();
       pose.orientation = Eigen::Quaternionf(poses[index].pose.orientation.w,
                                                poses[index].pose.orientation.x,
                                                poses[index].pose.orientation.y,
                                                poses[index].pose.orientation.z
                                               );
       pose.position = Eigen::Vector3f(poses[index].pose.position.x,
                                                poses[index].pose.position.y,
                                                poses[index].pose.position.z
                                               );
       populateTempGrid(grid, tempGrid, tempAcc,
  		                   clouds[index]->width, clouds[index]->height, clouds[index]->point_step,
  		                   &(clouds[index]->data[0]), pose, maxRange, logOddsMiss, logOddsHit, gridRes, frameTransform);
       integrateTempGrid(grid, tempGrid, gridAcc, logEmptyClampThres, logOccClampThres, logOddsThresMin, logOddsThresMax);
       IterType iter{grid->tree()};
       iter.setMaxDepth(IterType::LEAF_DEPTH);
   }
   handle_end = std::chrono::high_resolution_clock::now();
   delay = handle_end-handle_start;
   mapUpdateTime+=delay.count();
}

std::unique_ptr<octomap::OcTree> octomapCreateOctree(const float voxelSize = 0.1) {
    auto octree = std::make_unique<octomap::OcTree>(voxelSize);
    octree->setProbHit(0.95);
    octree->setProbMiss(0.35);
    octree->setClampingThresMin(0.12);
    octree->setClampingThresMax(0.97);
    return octree;
}

std::string convertNanosecondsSinceEpoc(long unsigned int nanoSinceEpoc)
{
  std::time_t msgTime = time_t((long int)(nanoSinceEpoc/1000000000));
  int nanoseconds = nanoSinceEpoc % 1000000000;
  char dateString[256];
  std::strftime(dateString, sizeof(dateString), "%F %T", std::localtime(&msgTime));
  return std::string(dateString)+"."+std::to_string(nanoseconds);
}

int main(int argc, char **argv) {
  std::cout << argc << std::endl;
    if(argc < 12){
        std::cout  << "pass bag, sensorTopic, sensorPoseTopic, sensorName, configFile, start, end, loopcount, sensorType, voxelizedOnly, and GPUOnly as arguments" << std::endl;
        exit(1);
    }

    const std::string bagPath(argv[1]);
    const std::string sensorTopic(argv[2]);
    const std::string sensorPoseTopic(argv[3]);
    std::string sensorName(argv[4]);
    std::string simConfigDir = argv[5];
    int indexTarget = atoi(argv[6]);
    int indexEnd = atoi(argv[7]);
    if(indexEnd==-1){
      indexEnd = MAX_INT;
    }
    int loopCount = atoi(argv[8]);
    int sensorType = atoi(argv[9]);
    int voxelizedOnly = atoi(argv[10]);
    std::cout << "1" << std::endl;
    int GPUOnly = atoi(argv[11]);
    std::cout << "2 " << std::endl;
    rosbag2_cpp::readers::SequentialReader reader;
    rosbag2_storage::StorageOptions storageOptions{};
    storageOptions.uri=bagPath;
    storageOptions.storage_id="sqlite3";

    rosbag2_cpp::ConverterOptions convertOptions{};
    convertOptions.input_serialization_format = "cdr";
    convertOptions.output_serialization_format = "cdr";
    reader.open(storageOptions, convertOptions);
    std::vector<rosbag2_storage::TopicMetadata> topic = reader.get_all_topics_and_types();
    std::map<std::string, std::string> nameTypeMap;
    for(auto t:topic)
    {
      std::cout << "meta name: " << t.name << std::endl;
      std::cout << "meta type: " << t.type << std::endl;
      std::cout << "meta serialization_format: " << t.serialization_format << std::endl;
      nameTypeMap[t.name]=t.type;
    }
    auto library_cloud = rosbag2_cpp::get_typesupport_library("sensor_msgs/msg/PointCloud2", "rosidl_typesupport_cpp");
    auto type_support_cloud = rosbag2_cpp::get_typesupport_handle("sensor_msgs/msg/PointCloud2", "rosidl_typesupport_cpp", library_cloud);
    auto library_pose = rosbag2_cpp::get_typesupport_library("geometry_msgs/msg/PoseStamped", "rosidl_typesupport_cpp");
    auto type_support_pose = rosbag2_cpp::get_typesupport_handle("geometry_msgs/msg/PoseStamped", "rosidl_typesupport_cpp", library_pose);
    std::vector<std::shared_ptr<sensor_msgs::msg::PointCloud2>> cloudPtrs;
    std::vector<geometry_msgs::msg::PoseStamped> poses;
    std::vector<geometry_msgs::msg::PoseStamped> matchedPoses;
    auto ros_message = std::make_shared<rosbag2_cpp::rosbag2_introspection_message_t>();
    rosbag2_cpp::SerializationFormatConverterFactory factory;
    std::unique_ptr<rosbag2_cpp::converter_interfaces::SerializationFormatDeserializer> cdr_deserializer;
    cdr_deserializer = factory.load_deserializer("cdr");
    std::shared_ptr<rosbag2_storage::SerializedBagMessage> serialized_message;
    ros_message->allocator = rcutils_get_default_allocator();
    int count  = 0;
    while(reader.has_next())
    {
      serialized_message = reader.read_next();
      ros_message->time_stamp = 0;
      ros_message->message = nullptr;
      if(strcmp(serialized_message->topic_name.c_str(),sensorPoseTopic.c_str())==0){
        if(count>= indexTarget){
          geometry_msgs::msg::PoseStamped pose;
          ros_message->message = &(pose);
          cdr_deserializer->deserialize(serialized_message,type_support_pose, ros_message);
          poses.push_back(pose);
        }
      }else if(strcmp(serialized_message->topic_name.c_str(),sensorTopic.c_str())==0){
        if(count >= indexTarget){
        std::shared_ptr<sensor_msgs::msg::PointCloud2> cloudPtr = std::make_shared<sensor_msgs::msg::PointCloud2>();
        ros_message->message = &(*cloudPtr);
        cdr_deserializer->deserialize(serialized_message,type_support_cloud, ros_message);
          if(!poses.empty()){
            cloudPtrs.push_back(cloudPtr);
            matchedPoses.push_back(poses.back());
          }
        }
        count += 1;
      }
      if(count>(indexEnd)){
        break;
      }
    }
    std::cout  << "cloudPtrs Size: " << cloudPtrs.size() << std::endl;
    indexEnd = cloudPtrs.size();
    openvdb::initialize();

    for(int i = 0; i < loopCount; i++){
    std::string voxelizedConfigStr = simConfigDir + "voxelizedConfig.txt";
    std::string parallelConfigStr = simConfigDir + "parallelConfig.txt";
    std::string vdbConfigStr = simConfigDir+"vdbConfig.txt";
    std::string sensorConfigStr = simConfigDir + sensorName + ".txt";
    std::chrono::duration<double, std::milli> delay;
    //define timers;
    int index;
    auto handle_start = std::chrono::high_resolution_clock::now();
    auto handle_end = std::chrono::high_resolution_clock::now();
    double handle_time=0;
    nanomap::Pose pose;
    // /*****************************************************************************/
        //voxelized, parallel update test;
        //Initialised required object
        std::shared_ptr<nanomap::config::Config> voxelizedConfig = std::make_shared<nanomap::config::Config>(nanomap::config::Config(voxelizedConfigStr));
        Eigen::Matrix<float, 3, 3> frameTransform = voxelizedConfig->sensorData(0)->sharedParameters()._frameTransform;

        if(sensorType == 0){
        std::shared_ptr<Map> voxelized_map_ptr = std::make_shared<Map>(Map(voxelizedConfig->mappingRes(), voxelizedConfig->probHitThres(), voxelizedConfig->probMissThres()));
        nanomap::manager::Manager voxelizedManager(voxelizedConfig);
        //Start timing and loop cloud insertion
        handle_start = std::chrono::high_resolution_clock::now();
        for(auto itr = cloudPtrs.begin(); itr != cloudPtrs.end(); itr++){

          index = std::distance(cloudPtrs.begin(), itr);
          pose.orientation = Eigen::Quaternionf(matchedPoses[index].pose.orientation.w,
                                                   matchedPoses[index].pose.orientation.x,
                                                   matchedPoses[index].pose.orientation.y,
                                                   matchedPoses[index].pose.orientation.z
                                                  );
          pose.position = Eigen::Vector3f(matchedPoses[index].pose.position.x,
                                                   matchedPoses[index].pose.position.y,
                                                   matchedPoses[index].pose.position.z
                                                  );
          voxelizedManager.insertPointCloud(sensorName, cloudPtrs[index]->width, cloudPtrs[index]->height ,
                                         cloudPtrs[index]->point_step , &(cloudPtrs[index]->data[0]), pose, voxelized_map_ptr);
        }
        //End loop and print time
        handle_end = std::chrono::high_resolution_clock::now();
        delay = handle_end-handle_start;
        handle_time = delay.count();
        voxelizedManager.printUpdateTime(index+1);
        std::cout << "voxelized nanomap time: " << handle_time << std::endl;
        std::cout << "voxelized nanomap time per loop: "<< handle_time/(index) << std::endl;
        //Save grid as octomap bt
        std::stringstream vox_ss;
        std::string vox_str;
        auto voxTree = octomapCreateOctree(voxelizedConfig->mappingRes());
        for (auto it = voxelized_map_ptr->occupiedGrid()->cbeginValueOn(); it; ++it) {
            auto idx = voxelized_map_ptr->occupiedGrid()->indexToWorld(it.getCoord());
            octomap::point3d worldP(idx.x()+(voxelizedConfig->mappingRes()/2), idx.y()+(voxelizedConfig->mappingRes()/2), idx.z()+(voxelizedConfig->mappingRes()/2));

            octomap::OcTreeKey key;
            voxTree->coordToKeyChecked(worldP, key);

            voxTree->updateNode(key, true);
        }
        vox_ss << "voxelized_gridres_" << voxelizedConfig->mappingRes() << ".bt" << std::endl;
        vox_ss >> vox_str;
        voxTree->writeBinary(vox_str);
        //Save grid as openvdb grid
        vox_ss.str("");
        vox_ss << "voxelized_gridres_" << voxelizedConfig->mappingRes() << ".vdb" << std::endl;
        vox_ss >> vox_str;
        openvdb::io::File(vox_str).write({voxelized_map_ptr->occupiedGrid()});
        //Close Handler
        voxelizedManager.closeHandler();

/*****************************************************************************/
    //Non voxelized, parallel update test;
    //Initialise required objects
    if(voxelizedOnly != 1){
    std::shared_ptr<nanomap::config::Config> parallelConfig = std::make_shared<nanomap::config::Config>(nanomap::config::Config(parallelConfigStr));

      std::shared_ptr<Map> parallel_map_ptr = std::make_shared<Map>(Map(parallelConfig->mappingRes(), parallelConfig->probHitThres(), parallelConfig->probMissThres()));
      nanomap::manager::Manager parallelManager(parallelConfig);
      //Start timing and loop cloud insertion
      handle_start = std::chrono::high_resolution_clock::now();
      for(auto itr = cloudPtrs.begin(); itr != cloudPtrs.end(); itr++){

        index = std::distance(cloudPtrs.begin(), itr);
        pose.orientation = Eigen::Quaternionf(matchedPoses[index].pose.orientation.w,
                                                 matchedPoses[index].pose.orientation.x,
                                                 matchedPoses[index].pose.orientation.y,
                                                 matchedPoses[index].pose.orientation.z
                                                );
        pose.position = Eigen::Vector3f(matchedPoses[index].pose.position.x,
                                                 matchedPoses[index].pose.position.y,
                                                 matchedPoses[index].pose.position.z
                                                );
        parallelManager.insertPointCloud(sensorName, cloudPtrs[index]->width, cloudPtrs[index]->height ,
                                       cloudPtrs[index]->point_step , &(cloudPtrs[index]->data[0]), pose, parallel_map_ptr);
      }
      //End loop and print time
      handle_end = std::chrono::high_resolution_clock::now();
      delay = handle_end-handle_start;
      handle_time = delay.count();
      parallelManager.printUpdateTime(index+1);
      std::cout << "parallel nanomap time: " << handle_time << std::endl;
      std::cout << "parallel nanomap time per loop: "<< handle_time/(index) << std::endl;

      //Save grid as octomap bt
      std::stringstream parallel_ss;
      std::string parallel_str;
      auto paraTree = octomapCreateOctree(parallelConfig->mappingRes());

      for (auto it = parallel_map_ptr->occupiedGrid()->cbeginValueOn(); it; ++it) {
          auto idx = parallel_map_ptr->occupiedGrid()->indexToWorld(it.getCoord());
          octomap::point3d worldP(idx.x()+(parallelConfig->mappingRes()/2), idx.y()+(parallelConfig->mappingRes()/2), idx.z()+(parallelConfig->mappingRes()/2));


          octomap::OcTreeKey key;
          paraTree->coordToKeyChecked(worldP, key);

          paraTree->updateNode(key, true);
      }
      parallel_ss << "parallel_gridres_" << parallelConfig->mappingRes() << ".bt" << std::endl;
      parallel_ss >> parallel_str;
      paraTree->writeBinary(parallel_str);
      //Save grid as openvdb grid
      parallel_ss.str("");
      parallel_ss << "parallel_gridres_" << parallelConfig->mappingRes() << ".vdb" << std::endl;
      parallel_ss >> parallel_str;
      openvdb::io::File(parallel_str).write({parallel_map_ptr->occupiedGrid()});
      //Close Handler
      parallelManager.closeHandler();
    }
  }
    if(GPUOnly == 0){
/********************************hostLaserVoxelCount()**********************************************/
    //cpu raycast, serial voxel update.
    float occClampThres =  0.97;
    float emptyClampThres = 0.12;

    float mappingRes, oddsMiss, oddsHit, oddsThresMin, oddsThresMax, maxRange;
    loadVDBConfig(vdbConfigStr, sensorConfigStr, mappingRes, oddsMiss, oddsHit, oddsThresMin, oddsThresMax, maxRange);
    float mapUpdateTime = 0.0;
    float cloudProcessTime = 0.0;
    index = indexEnd-indexTarget;
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(mappingRes));
    grid->setGridClass(openvdb::GRID_LEVEL_SET);
    grid->setName("vdbTestGrid");
    openvdb::FloatGrid::Accessor gridAcc = grid->getAccessor();
    handle_start = std::chrono::high_resolution_clock::now();
    vdbMappingTest( cloudPtrs,
                    matchedPoses,
                    grid,
                    gridAcc,
                    mappingRes,
                    maxRange,
                    oddsMiss,
                    oddsHit,
                    oddsThresMin,
                    oddsThresMax,
                    occClampThres,
                    emptyClampThres,
                    mapUpdateTime,
                    cloudProcessTime,
                    frameTransform,
                    indexTarget,
                    indexEnd);
    handle_end = std::chrono::high_resolution_clock::now();
    delay = handle_end-handle_start;
    handle_time = delay.count();
    std::cout << "vdb time: " << handle_time << std::endl;
    std::cout << "vdb time per loop: "<< handle_time/(index) << std::endl;

    std::stringstream vdb_ss;
    std::string vdb_str;
    auto vdbTree = octomapCreateOctree(mappingRes);
    for (auto it = grid->cbeginValueOn(); it; ++it) {
        auto idx = grid->indexToWorld(it.getCoord());
        octomap::point3d worldP(idx.x()+(mappingRes/2), idx.y()+(mappingRes/2), idx.z()+(mappingRes/2));

        octomap::OcTreeKey key;
        vdbTree->coordToKeyChecked(worldP, key);

        vdbTree->updateNode(key, true);
    }
    vdb_ss << "vdb_gridres_" << mappingRes << ".bt" << std::endl;
    vdb_ss >> vdb_str;
    vdbTree->writeBinary(vdb_str);


    vdb_ss << "vdb_gridres_" << mappingRes << ".vdb" << std::endl;
    vdb_ss>>vdb_str;
    openvdb::io::File(vdb_str).write({grid});

// /*****************************************************************************/
//Test octree, parallel bulk scan operation
    auto octree = octomapCreateOctree(mappingRes);
    handle_start = std::chrono::high_resolution_clock::now();
    octomapBuildMapBulkScan(frameTransform,cloudPtrs, matchedPoses, maxRange, octree.get(), false);
    handle_end = std::chrono::high_resolution_clock::now();
    delay = handle_end - handle_start;
    handle_time = delay.count();
    std::cout << "octomap time: " << handle_time << std::endl;
    std::cout << "octomap time per loop: " << handle_time/(index) << std::endl;
    std::stringstream octo_ss;
    std::string octo_str;
    octo_ss << "octomap_gridres_" << mappingRes << ".bt" << std::endl;
    octo_ss >> octo_str;
    octree->writeBinary(octo_str);

    auto octreeDiscretized = octomapCreateOctree(mappingRes);
    handle_start = std::chrono::high_resolution_clock::now();
    octomapBuildMapBulkScan(frameTransform,cloudPtrs, matchedPoses, maxRange, octreeDiscretized.get(), true);
    handle_end = std::chrono::high_resolution_clock::now();
    delay = handle_end - handle_start;
    handle_time = delay.count();
    std::cout << "octomap discretized time: " << handle_time << std::endl;
    std::cout << "octomap discretized time per loop: " << handle_time/(index) << std::endl;
    std::stringstream octo_disc_ss;
    std::string octo_disc_str;
    octo_disc_ss << "discrete_octomap_gridres_" << mappingRes << ".bt" << std::endl;
    octo_disc_ss >> octo_disc_str;
    octreeDiscretized->writeBinary(octo_disc_str);
  }
  }
}
