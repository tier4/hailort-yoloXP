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

#include "config_parser.hpp"
#include <assert.h>
#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>

DEFINE_string(hef, "hefs/yolox_dla-s-elan-960x960-T4.hef",
              "HEF Path, "
              "HEF Path");

DEFINE_bool(dont_show, false,
	    "[Optional] Flag to off screen");

DEFINE_string(d, "",
              "Directory Path, "
              "Directory Path");

DEFINE_string(v, "",
              "Video Path, "
              "Video Path");

DEFINE_int64(cam_id, -1, "Camera ID");

DEFINE_string(dump, "not-specified",
              "[OPTIONAL] Path to dump predictions for mAP calculation");

DEFINE_uint64(c, 8, "[OPTIONAL] num of classes for the inference engine.");

DEFINE_string(rgb, "not-specified",
              "[OPTIONAL] Path to pre-generated calibration table. If flag is not set, a new calib "
              "table <network-type>-<precision>-calibration.table will be generated");

DEFINE_string(names, "../data/t4.names",
              "[OPTIONAL] Path to pre-generated calibration table. If flag is not set, a new calib "
              "table <network-type>-<precision>-calibration.table will be generated");

DEFINE_double(thresh, 0.3, "[OPTIONAL] thresh");

DEFINE_double(nmsThresh, 0.45, "[OPTIONAL] thresh");

DEFINE_bool(save_detections, false,
            "[OPTIONAL] Flag to save images overlayed with objects detected.");
DEFINE_string(save_detections_path, "outputs/",
              "[OPTIONAL] Path where the images overlayed with bounding boxes are to be saved");

DEFINE_int64(reduce_ratio, 4, "[OPTIONAL] The reduce ratio of the original video when show the video");

std::string
get_hef_path(void)
{
  return FLAGS_hef;
}

std::string
get_directory_path(void)
{
  return FLAGS_d;
}

int
get_camera_id(void)
{
  return FLAGS_cam_id;
}

std::string
get_video_path(void)
{
  return FLAGS_v;
}

bool
is_dont_show(void)
{
  return FLAGS_dont_show;
}

int
get_classes(void)
{
  return FLAGS_c;
}

static void leftTrim(std::string& s)
{
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

static std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}

static bool fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

static std::vector<std::string> loadListFromTextFile(const std::string filename)
{
  assert(fileExists(filename, true));
    std::vector<std::string> list;

    std::ifstream f(filename);
    if (!f)
    {
        std::cout << "failed to open " << filename;
        assert(0);
    }

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;

        else
            list.push_back(trim(line));
    }

    return list;
}

std::vector<std::vector<int>>
get_colormap(void)
{
  std::string filename = FLAGS_rgb;
  std::vector<std::vector<int>> colormap;
  if (filename != "not-specified") {
    std::vector<std::string> color_list = loadListFromTextFile(filename);    
    for (int i = 0; i < (int)color_list.size(); i++) {
      std::string colormapString = color_list[i];
      std::vector<int> rgb;
      while (!colormapString.empty()) {
	size_t npos = colormapString.find_first_of(',');
	if (npos != std::string::npos) {
	  int colormap = (int)std::stoi(trim(colormapString.substr(0, npos)));
	  rgb.push_back(colormap);
	  colormapString.erase(0, npos + 1);
	} else {
	  int colormap = (int)std::stoi(trim(colormapString));
	  rgb.push_back(colormap);
	  break;
	}      
      }
      colormap.push_back(rgb);
    }
  }
  return colormap;
}

std::vector<std::string>
get_names(void)
{
  std::string filename = FLAGS_names;
  std::vector<std::string> names;
  if (filename != "not-specified") {
    names = loadListFromTextFile(filename);    
  }
  return names;
}


double
get_score_thresh(void)
{
  return FLAGS_thresh;
}

double
get_nms_thresh(void)
{
  return FLAGS_nmsThresh;
}

int
get_reduce_ratio(void)
{
  return FLAGS_reduce_ratio;
}


static bool isFlagDefault(std::string flag) { return flag == "not-specified" ? true : false; }

bool getSaveDetections()
{
  if (FLAGS_save_detections)
    assert(!isFlagDefault(FLAGS_save_detections_path)
	   && "save_detections path has to be set if save_detections is set to true");
  return FLAGS_save_detections;
}

std::string getSaveDetectionsPath() { return FLAGS_save_detections_path; }

std::string
get_dump_path(void)
{
  return FLAGS_dump;
}
