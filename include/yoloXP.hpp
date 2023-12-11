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

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>
#include "common.hpp"

namespace yoloXP
{
  struct Object
  {
    int32_t x_offset;
    int32_t y_offset;
    int32_t height;
    int32_t width;
    float score;
    int32_t type;
  };

  using ObjectArray = std::vector<Object>;
  using ObjectArrays = std::vector<ObjectArray>;
  struct GridAndStride
  {
    int grid0;
    int grid1;
    int stride;
  };

  /**
   * @class YoloXP
   * @brief YoloXP Inference
   */ 
  class YoloXP
  {
  public:

    /**
     * @brief Construct YoloXP
     * @param[in] output_strides stride for output grid
     * @param[in] num_classes classes for outputs
     * @param[in] thresh score thresh
     * @param[in] nms_thresh nms thresh
     */    
    YoloXP(const std::vector<int> &output_strides, const int num_classes, const float thresh, const float nms_thresh);    
    
    ~YoloXP();

    /**
     * @brief run preprcess including resizing and letterbox on CPU
     * @param[in] image input image
     * @warning NHWC2NCHW and toFloat is not required for HailoRT
     */            
    cv::Mat
    preprocess(const cv::Mat & image,
	       const int input_height,
	       const int input_width);

    /**
     * @brief decode yoloXP outputs
     * @param[in] output_stream hailoRT output_stream 
     * @param[in] input_stream hailoRT input_stream 
     * @param[in] prob raw feature
     * @param[out] objects bounding boxes
     * @param[in] img_size input image size
     */                
    void
    decodeOutputs(const OutputVStream &output_stream, const InputVStream &input_stream, const float * prob, ObjectArray & objects, const cv::Size & img_size);

    /**
     * @brief draw bounding boxex
     * @param[in] img input image
     * @param[in] objects bounding boxes
     * @param[in] colormap colormap
     */                    
    void
    drawBBox(cv::Mat &img, ObjectArray &objects, const uint8_t *colormap);

    cv::Mat
    getMask(const uint8_t *prob, int classes, int width, int height, int out_w, int out_h, const unsigned char*colormap);
    
    
  protected:
    std::vector<int> m_output_strides;
    int m_num_classes;
    float m_thresh;
    float m_nms_thresh;
    
  private:
    void
    generateGridsAndStride(
			   const int target_w, const int target_h, std::vector<int> strides,
			   std::vector<GridAndStride> & grid_strides);
    
    void
    generateYoloXPProposals(
			   const std::vector<GridAndStride> grid_strides, const float * feat_blob, const float prob_threshold,
			   ObjectArray & objects, const int numClasses, const hailo_quant_info_t quant_info);
    void
    qsortDescentInplace(ObjectArray & faceobjects, int left, int right);

    inline void qsortDescentInplace(ObjectArray & objects)
    {
      if (objects.empty()) {
	return;
      }
      qsortDescentInplace(objects, 0, objects.size() - 1);
    }

    inline float intersectionArea(const Object & a, const Object & b)
    {
      cv::Rect a_rect(a.x_offset, a.y_offset, a.width, a.height);
      cv::Rect b_rect(b.x_offset, b.y_offset, b.width, b.height);
      cv::Rect_<float> inter = a_rect & b_rect;
      return inter.area();
    }

    void
    nmsSortedBboxes(
		    const ObjectArray & faceobjects, std::vector<int> & picked, float nms_threshold);

  };    
}
