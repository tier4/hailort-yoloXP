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
/**
 * Copyright (c) 2020-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/
/**
 * @file vstreams_example
 * This example demonstrates using virtual streams over c++
 **/

#include "common.hpp"
#include "utils.hpp"

namespace hailortCommon
{
  HrtCommon::HrtCommon()
  {
  }

  HrtCommon::~HrtCommon()
  {
  }  
  
  void writeAll(InputVStream &input, hailo_status &status, unsigned char *preprocessed)
  {
    for (size_t i = 0; i < FRAMES_COUNT; i++) {
      status = input.write(MemoryView(preprocessed, input.get_frame_size()));      
      if (HAILO_SUCCESS != status) {
	return;
      }
    }

    // Flushing is not mandatory here
    status = input.flush();
    if (HAILO_SUCCESS != status) {
      std::cerr << "Failed flushing input vstream" << std::endl;
      return;
    }

    status = HAILO_SUCCESS;
    return;
  }

  void readAll(OutputVStream &output, hailo_status &status,  std::vector<uint8_t> &data)
  {
    for (size_t i = 0; i < FRAMES_COUNT; i++) {
      status = output.read(MemoryView(data.data(), data.size()));
      if (HAILO_SUCCESS != status) {
	return;
      }
    }
    status = HAILO_SUCCESS;
    return;
  }

  void HrtCommon::printOutputTensorInfos(const hailo_vstream_info_t info)
  {
    auto shape = info.shape;
    auto width = shape.width;
    auto height = shape.height;    
    auto feature = shape.features;
    std::cout << "-> " << info.name << " :"<< width << "x" <<  height << "x" << feature<< std::endl;
    hailo_quant_info_t quant_info = info.quant_info;
    float qp_scale = quant_info.qp_scale;
    float qp_zp = quant_info.qp_zp;
    std::cout << "--> " << "qp_scale=" << qp_scale << " qp_zp=" << qp_zp << " ";
    if (info.format.type == HAILO_FORMAT_TYPE_UINT8) {
      std::cout << "HAILO_FORMAT_TYPE_UINT8";
    } else if (info.format.type == HAILO_FORMAT_TYPE_UINT16) {
      std::cout << "HAILO_FORMAT_TYPE_UINT16";
    }
    std::cout << std::endl;
  }
  
  
  float HrtCommon::dequant(void *qv,  const float qp_scale, const float qp_zp, hailo_format_type_t format)
  {
    float dqv=0.0;
    if (format == HAILO_FORMAT_TYPE_UINT16) {
      dqv = dequantInt16(*(uint16_t *)qv, qp_scale, qp_zp);
    } else if (format == HAILO_FORMAT_TYPE_UINT8) {
      dqv = dequantInt8(*(uint8_t *)qv, qp_scale, qp_zp);
    } else {
      std::cout << "Warn : Unsupport Format" << std::endl;
    }
    return dqv;
  }

  void HrtCommon::printMeasurementsResults(Device &device, const hailo_power_measurement_data_t &result, hailo_power_measurement_types_t type)
  {
    auto id = device.get_dev_id();

    auto type_str = (type == HAILO_POWER_MEASUREMENT_TYPES__POWER) ? "Power measurement" :
      "Current measurement";

    std::cout << "Device" << std::string(id) << ":" << std::endl;
    std::cout << "  " << type_str << std::endl;
    std::cout << "    Minimum value: " << result.min_value << MEASUREMENT_UNITS(type) << std::endl;
    std::cout << "    Average value: " << result.average_value << MEASUREMENT_UNITS(type) << std::endl;
    std::cout << "    Maximum value: " << result.max_value << MEASUREMENT_UNITS(type) << std::endl;
  }

  Expected<std::shared_ptr<ConfiguredNetworkGroup>>
  HrtCommon::configureNetworkGroup(VDevice &vdevice, std::string &hef_file)
  {
    auto hef = Hef::create(hef_file);
    if (!hef) {
      return make_unexpected(hef.status());
    }

    auto configure_params = vdevice.create_configure_params(hef.value());
    if (!configure_params) {
      return make_unexpected(configure_params.status());
    }

    auto network_groups = vdevice.configure(hef.value(), configure_params.value());
    if (!network_groups) {
      return make_unexpected(network_groups.status());
    }

    if (1 != network_groups->size()) {
      std::cerr << "Invalid amount of network groups" << std::endl;
      return make_unexpected(HAILO_INTERNAL_FAILURE);
    }
    return std::move(network_groups->at(0));
  }

  hailo_status
  HrtCommon::infer(std::vector<InputVStream> &input_streams, std::vector<OutputVStream> &output_streams, unsigned char *data, std::vector<std::vector<uint8_t>> &results)
  {
    hailo_status status = HAILO_SUCCESS; // Success oriented
    hailo_status input_status[MAX_LAYER_EDGES] = {HAILO_UNINITIALIZED};
    hailo_status output_status[MAX_LAYER_EDGES] = {HAILO_UNINITIALIZED};
    std::unique_ptr<std::thread> input_threads[MAX_LAYER_EDGES];
    std::unique_ptr<std::thread> output_threads[MAX_LAYER_EDGES];
    size_t input_thread_index = 0;
    size_t output_thread_index = 0;
    // Create read threads

    for (output_thread_index = 0 ; output_thread_index < output_streams.size(); output_thread_index++) {
      output_threads[output_thread_index] = std::make_unique<std::thread>(readAll,
									  std::ref(output_streams[output_thread_index]), std::ref(output_status[output_thread_index]), std::ref(results[output_thread_index]));
    }

    // Create write threads
    for (input_thread_index = 0 ; input_thread_index < input_streams.size(); input_thread_index++) {
      input_threads[input_thread_index] = std::make_unique<std::thread>(writeAll,
									std::ref(input_streams[input_thread_index]), std::ref(input_status[input_thread_index]), data);
    }

    // Join write threads
    for (size_t i = 0; i < input_thread_index; i++) {
      input_threads[i]->join();
      if (HAILO_SUCCESS != input_status[i]) {
	status = input_status[i];
      }
    }

    // Join read threads
    for (size_t i = 0; i < output_thread_index; i++) {
      output_threads[i]->join();
      if (HAILO_SUCCESS != output_status[i]) {
	status = output_status[i];
      }
    }

    if (HAILO_SUCCESS == status) {
      //std::cout << "Inference finished successfully" << std::endl;
    }

    return status;
  }
}
