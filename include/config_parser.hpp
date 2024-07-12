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

#include <memory>
#include <string>
#include <gflags/gflags.h>

extern std::string
get_hef_path(void);

extern std::string
get_directory_path(void);

extern std::string
get_video_path(void);

extern int
get_camera_id(void);
  
extern std::string
get_precision(void);

extern bool
is_dont_show(void);

extern int
get_classes(void);

extern std::vector<std::vector<int>>
get_colormap(void);

extern std::vector<std::string>
get_names(void);

extern double
get_score_thresh(void);

extern double
get_nms_thresh(void);

extern int
get_reduce_ratio(void);

extern bool
getSaveDetections();

extern std::string
getSaveDetectionsPath();

extern std::string
get_dump_path(void);
