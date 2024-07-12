// Copyright 2023 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils.hpp"
#include "yoloXP.hpp"
#include "colormap.hpp"
#include "class_timer.hpp"
#include "config_parser.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <boost/filesystem.hpp>
#include <experimental/filesystem>


namespace fs = boost::filesystem;
template <typename ... Args>
std::string format(const std::string& fmt, Args ... args )
{
  size_t len = std::snprintf( nullptr, 0, fmt.c_str(), args ... );
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ... );
  return std::string(&buf[0], &buf[0] + len);
}

void
write_prediction(std::string dumpPath, std::string filename, std::vector<std::string> names, yoloXP::ObjectArray &objects, int width, int height)
{
  int pos = filename.find_last_of(".");
  std::string body = filename.substr(0, pos);
  std::string dstName = body + ".txt";
  std::ofstream writing_file;
  fs::path p = dumpPath;
  fs::create_directory(p);  
  p.append(dstName);
  writing_file.open(p.string(), std::ios::out);
  for (const auto & object : objects) {
    const auto left = object.x_offset;
    const auto top = object.y_offset;
    const auto right = clamp(left + object.width, 0, width);
    const auto bottom = clamp(top + object.height, 0, height);
    const auto id = object.type;  
    std::string writing_text = format("%s %f %d %d %d %d", names[id].c_str(), object.score, left, top, (int)right, (int)bottom);
    //std::cout << writing_text  << std::endl;
    writing_file << writing_text << std::endl;
  }
  writing_file.close();
}

void doPreprocessFromCap(std::shared_ptr<yoloXP::YoloXP> yoloXP, cv::VideoCapture &cap, cv::Mat &src, cv::Mat &input, int input_w, int input_h)
{
  Timer timer;
  timer.reset();
  cap >> src;
  if (src.empty() == false) {
    input = yoloXP->preprocess(src, input_w, input_h);
  }
  timer.out("Preprocess");
}

void infer(std::shared_ptr<hailortCommon::HrtCommon> hrtCommon, std::vector<InputVStream> &input_streams, std::vector<OutputVStream> &output_streams, unsigned char *data, std::vector<std::vector<uint8_t>> &results, float *inftime)
{
  Timer timer;
  timer.reset();  
  hailo_status status = hrtCommon->infer(input_streams, output_streams, data, results);
  if (HAILO_SUCCESS != status) {
    std::cerr << "Inference failed "  << status << std::endl;
  }
  *inftime = timer.out("Inference");
}

void doPostprocess(std::shared_ptr<hailortCommon::HrtCommon> hrtCommon, std::shared_ptr<yoloXP::YoloXP> yoloXP, std::vector<InputVStream> &input_streams, std::vector<OutputVStream> &output_streams, std::vector<std::vector<uint8_t>> &results, std::vector<std::vector<float>> &f_results, cv::Mat &src, float elapsed, float inftime, float max_power)
{
  Timer timer;
  timer.reset();  
  int numClasses = get_classes();
  for (int output_index = 0 ; output_index < (int)output_streams.size(); output_index++) {
    yoloXP::ObjectArray object_array;
    const int chan = (4+1+numClasses);      
    auto info = output_streams[output_index].get_info();
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;
    auto feature = shape.features;
    hailo_quant_info_t quant_info = info.quant_info;
    float qp_scale = quant_info.qp_scale;
    float qp_zp = quant_info.qp_zp;
    auto size = output_streams[output_index].get_frame_size();
    std::vector<float> data(size);
    if (height == 1) {  //YOLOXP Layer
      //Post-Process
      //ToDo : Remove last layer (Transpose) from ONNX for optimization in HailoRT
      for (int i = 0; i < (int)size; i++) {
        int xy = i % feature;
        int c = i / feature;
        int index = xy * chan + c;
        f_results[output_index][index] = hrtCommon->dequant(&results[output_index][i], qp_scale, qp_zp, info.format.type);	
      }
      
      yoloXP->decodeOutputs(std::ref(output_streams[output_index]), std::ref(input_streams[0]), f_results[output_index].data(), object_array, src.size());
      yoloXP->drawBBox(src, object_array, colormap);
    } else { //Segmentation
      const float scale = std::min(width / float(src.cols), height / float(src.rows));
      int out_w = (int)(src.cols*scale);
      int out_h = (int)(src.rows*scale);
      Timer timer;
      auto cmask = yoloXP->getMask(results[output_index].data(), feature, width, height, out_w, out_h, semseg_colormap);      
      cv::Mat resized;
      cv::resize(cmask, resized, cv::Size(src.cols, src.rows), 0, 0, cv::INTER_NEAREST);
      cv::addWeighted(src, 1.0, resized, 0.5, 0.0, src);
      //      cv::imshow("mask", cmask);
    }
  }
  float fps = 1000/elapsed;
  cv::putText(src, "FPS :" + format(fps, 4) , cv::Point(20, src.rows-60), 0, 1, cv::Scalar(255, 255,255), 1);
  cv::putText(src, "DNN Time :" + format(inftime, 4) , cv::Point(240, src.rows-60), 0, 1, cv::Scalar(255, 255,255), 1);      
  cv::putText(src, "Peak Power :" + format(max_power, 4) + "W", cv::Point(20, src.rows-20), 0, 1, cv::Scalar(255, 255,255), 1);

  cv::imshow("hailort_inference", src);
  if (cv::waitKey(1) == 'q');
  timer.out("Postprocess");
}


int
main(int argc, char* argv[])
{
  std::vector<std::vector<uint8_t>> results;
  std::vector<std::vector<float>> f_results;
  int output_index;
  // cv::VideoCapture video;
  cv::Mat src;    
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string videoName = get_video_path();
  const int cam_id = get_camera_id();  
  std::string hef_path = get_hef_path();
  float nmsThresh = get_nms_thresh();
  int numClasses = get_classes();
  float thresh = get_score_thresh();
  int reduce_ratio = get_reduce_ratio();
  std::vector<int> output_strides = {8, 16, 32};

  cv::VideoCapture video;

  auto hrtCommon = std::make_shared<hailortCommon::HrtCommon>();
  
  Expected<std::unique_ptr<VDevice>> vdevice = VDevice::create();
  if (vdevice) {
    std::cerr << "Failed create vdevice, status = " << vdevice.status() << std::endl;
  }
  
  Expected<std::vector<std::reference_wrapper<Device>>> physical_devices = vdevice.value()->get_physical_devices();
  if (!physical_devices) {
    std::cerr << "Failed to get physical devices" << std::endl;
  }
  
  auto network_group = hrtCommon->configureNetworkGroup(*vdevice.value(), hef_path);
  if (!network_group) {
    std::cerr << "Failed to configure network group " << hef_path << std::endl;
    return network_group.status();
  }

  Expected<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams = VStreamsBuilder::create_vstreams(*network_group.value(), QUANTIZED, FORMAT_TYPE);
  if (!vstreams) {
    std::cerr << "Failed creating vstreams " << vstreams.status() << std::endl;
    return vstreams.status();
  }

  if (vstreams->first.size() > MAX_LAYER_EDGES || vstreams->second.size() > MAX_LAYER_EDGES) {
    std::cerr << "Trying to infer network with too many input/output virtual streams, Maximum amount is " <<
      MAX_LAYER_EDGES << " (either change HEF or change the definition of MAX_LAYER_EDGES)"<< std::endl;
    return HAILO_INVALID_OPERATION;
  }
  
  for (output_index = 0 ; output_index < (int)vstreams->second.size(); output_index++) {
    auto size = vstreams->second[output_index].get_frame_size();
    auto info = vstreams->second[output_index].get_info();
    std::vector<uint8_t> data(size);
    results.emplace_back(data);      
    std::vector<float> f_data(size*4);
    f_results.emplace_back(f_data);
    //Currently support fully INT8 Outputs
    assert(info.format.type == HAILO_FORMAT_TYPE_UINT8);
  }

  for (auto &physical_device : physical_devices.value()) {
    auto measurement_type = HAILO_POWER_MEASUREMENT_TYPES__POWER;
    auto p_status = physical_device.get().stop_power_measurement();
    p_status = physical_device.get().set_power_measurement(MEASUREMENT_BUFFER_INDEX, DVM_OPTION, measurement_type);    
    p_status = physical_device.get().start_power_measurement(AVERAGE_FACTOR, SAMPLING_PERIOD);
    if (HAILO_SUCCESS != p_status) {
      std::cerr << "Failed to start measurement" << std::endl;
      return p_status;
    }    
  }

  auto input_info = vstreams->first[0].get_info();  
  const int input_w = input_info.shape.width;    
  const int input_h = input_info.shape.height;
  auto yoloXP = std::make_shared<yoloXP::YoloXP>(output_strides, numClasses, thresh, nmsThresh);
  float elapsed = 0.0;
  float inftime = 0.0;    


  if (videoName != "" || cam_id != -1) { 
    if (cam_id != -1) {
      video.open(cam_id);
    } else {
      video.open(videoName);
    }

    // Get the width and height of the video frames
    int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Print the frame size
    std::cout << "Frame Width: " << frame_width << std::endl;
    std::cout << "Frame Height: " << frame_height << std::endl;

    // Create a window to display the video
    cv::namedWindow("hailort_inference", cv::WINDOW_NORMAL);
    cv::resizeWindow("hailort_inference", int(frame_width / reduce_ratio), int(frame_height / reduce_ratio));

    cv::Mat input;
    video >> src;
    if (!src.empty()) {
      input = yoloXP->preprocess(src, input_w, input_h);
    }
    infer(hrtCommon, std::ref(vstreams->first),std::ref(vstreams->second), std::ref(input.data), std::ref(results), &inftime);
    cv::Mat vis = src.clone();
    float max_power = 0.0;

    while (1) {
      Timer timer;
      if (src.empty() == true) break;
      timer.reset();
      cv::Mat in = input.clone();
      vis = src.clone();
      std::vector<std::vector<uint8_t>> prev(results.begin(), results.end());
      std::thread preprocess_thread(doPreprocessFromCap, yoloXP, std::ref(video), std::ref(src), std::ref(input), input_w, input_h);
      std::thread inference_thread(infer, hrtCommon, std::ref(vstreams->first),std::ref(vstreams->second), std::ref(in.data), std::ref(results), &inftime);
      std::thread postprocess_thread(doPostprocess,  hrtCommon, yoloXP, std::ref(vstreams->first), std::ref(vstreams->second), std::ref(prev),  std::ref(f_results),  std::ref(vis), elapsed, inftime, max_power);
      postprocess_thread.join();      
      preprocess_thread.join();      
      inference_thread.join();
      elapsed = timer.out("Total");
      for (auto &physical_device : physical_devices.value()) {
        auto measurement_result = physical_device.get().get_power_measurement(MEASUREMENT_BUFFER_INDEX, true);
        max_power = measurement_result.value().max_value;
      }          
    }
  } else {
    std::string directory = get_directory_path();
    const std::string dumpPath = get_dump_path();
    for (const auto & file : std::experimental::filesystem::directory_iterator(directory)) {
      std::cout << "Read ... " << file.path() << std::endl;
      src = cv::imread(file.path(), cv::IMREAD_UNCHANGED);	  
      if (src.empty() == true) break;
      cv::Mat input = yoloXP->preprocess(src, input_w, input_h);
      float max_power = 0.0;
      Timer timer;
      infer(hrtCommon, std::ref(vstreams->first),std::ref(vstreams->second), std::ref(input.data), std::ref(results), &inftime);      
      elapsed = timer.out("inference");
      for (auto &physical_device : physical_devices.value()) {
        auto measurement_result = physical_device.get().get_power_measurement(MEASUREMENT_BUFFER_INDEX, true);
        max_power = measurement_result.value().max_value;
      }    
      doPostprocess(hrtCommon, yoloXP, std::ref(vstreams->first), std::ref(vstreams->second), std::ref(results),  std::ref(f_results),  std::ref(src), elapsed, inftime, max_power);
    }
  }
  
  for (auto &physical_device : physical_devices.value()) {  
    auto p_status = physical_device.get().stop_power_measurement();
    if (HAILO_SUCCESS != p_status) {
      std::cerr << "Failed to stop measurement" << std::endl;
      return p_status;
    }        
  }
  
  return HAILO_SUCCESS;
}
