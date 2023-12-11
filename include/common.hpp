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

#include "hailo/hailort.hpp"
#include <thread>
using namespace hailort;

#define SAMPLING_PERIOD (HAILO_SAMPLING_PERIOD_332US)
#define AVERAGE_FACTOR (HAILO_AVERAGE_FACTOR_16)
#define DVM_OPTION (HAILO_DVM_OPTIONS_AUTO) // For current measurement over EVB - pass DVM explicitly (see hailo_dvm_options_t)
#define MEASUREMENT_BUFFER_INDEX (HAILO_MEASUREMENT_BUFFER_INDEX_0)

#define MEASUREMENT_UNITS(__type) \
    ((HAILO_POWER_MEASUREMENT_TYPES__POWER == __type) ? ("W") : ("mA"))

#define USAGE_ERROR_MSG ("Args parsing error.\nUsage: power_measurement_example [power / current]\n" \
    "* power   - measure power consumption in W\n" \
    "* current - measure current in mA\n")

const std::string POWER_ARG = "power";
const std::string CURRENT_ARG = "current";
const std::chrono::seconds MEASUREMENTS_DURATION_SECS(5);

constexpr size_t FRAMES_COUNT = 1;
constexpr bool QUANTIZED = true;
constexpr hailo_format_type_t FORMAT_TYPE = HAILO_FORMAT_TYPE_AUTO;
constexpr size_t MAX_LAYER_EDGES = 16;

extern void
writeAll(InputVStream &input, hailo_status &status, unsigned char *preprocessed);

extern void
readAll(OutputVStream &output, hailo_status &status,  std::vector<uint8_t> &data);

namespace hailortCommon
{
  class HrtCommon
  {
  public:
    HrtCommon();

    ~HrtCommon();    

    void printOutputTensorInfos(const hailo_vstream_info_t info);
    
    float dequant(void *qv,  const float qp_scale, const float qp_zp, hailo_format_type_t format);

    void printMeasurementsResults(Device &device, const hailo_power_measurement_data_t &result, hailo_power_measurement_types_t type);

    Expected<std::shared_ptr<ConfiguredNetworkGroup>>
    configureNetworkGroup(VDevice &vdevice, std::string &hef_file);

    hailo_status
    infer(std::vector<InputVStream> &input_streams, std::vector<OutputVStream> &output_streams, unsigned char *data, std::vector<std::vector<uint8_t>> &results);
  protected:    
  private:
  };
}
