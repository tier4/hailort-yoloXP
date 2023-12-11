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

namespace yoloXP
{
  YoloXP::YoloXP(const std::vector<int> &output_strides, const int num_classes, const float thresh, const float nms_thresh) :
    m_output_strides(output_strides),
    m_num_classes(num_classes),
    m_thresh(thresh),
    m_nms_thresh(nms_thresh)
  {

  }

  YoloXP::~YoloXP()
  {

  }  
  
  cv::Mat
  YoloXP::preprocess(const cv::Mat & image,
	     const int input_height,
	     const int input_width)
  {
    cv::Mat dst_image;
    const float scale = std::min(input_width / (float)image.cols, input_height / (float)image.rows);
    const auto scale_size = cv::Size(image.cols * scale, image.rows * scale);
    cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_LINEAR);
    const auto bottom = input_height - dst_image.rows;
    const auto right = input_width - dst_image.cols;
    copyMakeBorder(dst_image, dst_image, 0, bottom, 0, right, cv::BORDER_CONSTANT, {114, 114, 114});  
    //NHWC format
    return dst_image;
  }

  void
  YoloXP::generateGridsAndStride(
			 const int target_w, const int target_h, std::vector<int> strides,
			 std::vector<GridAndStride> & grid_strides)
  {
    for (auto stride : strides) {
      int num_grid_w = target_w / stride;
      int num_grid_h = target_h / stride;
      for (int g1 = 0; g1 < num_grid_h; g1++) {
	for (int g0 = 0; g0 < num_grid_w; g0++) {
	  grid_strides.push_back(GridAndStride{g0, g1, stride});
	}
      }
    }
  }

  void
  YoloXP::generateYoloXPProposals(
			 std::vector<GridAndStride> grid_strides, const float * feat_blob, float prob_threshold,
			 ObjectArray & objects, const int numClasses, hailo_quant_info_t quant_info)
  {
    const int num_anchors = grid_strides.size();
    //const float qp_scale = quant_info.qp_scale;
    //const float qp_zp = quant_info.qp_zp;
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
      const int grid0 = grid_strides[anchor_idx].grid0;
      const int grid1 = grid_strides[anchor_idx].grid1;
      const int stride = grid_strides[anchor_idx].stride;

      //outputs : NCHW
      //original : NHWC
      const int basic_pos = anchor_idx * (numClasses + 5);
      // yoloXP/models/yolo_head.py decode logic
      // To apply this logic, YOLOXP head must output raw value
      // (i.e., `decode_in_inference` should be False)
      float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
      float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
      /*
      //exp is complex for embedded processors
      float w = exp(feat_blob[basic_pos + 2]) * stride;
      float h = exp(feat_blob[basic_pos + 3]) * stride;
      float x0 = x_center - w * 0.5f;
      float y0 = y_center - h * 0.5f;
      */
      float box_objectness = feat_blob[basic_pos + 4];
      for (int class_idx = 0; class_idx < numClasses; class_idx++) {
	float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
	float box_prob = box_objectness * box_cls_score;
	if (box_prob > prob_threshold) {
	  Object obj;
	  //On-demand applying for exp 
	  float w = exp(feat_blob[basic_pos + 2]) * stride;
	  float h = exp(feat_blob[basic_pos + 3]) * stride;
	  float x0 = x_center - w * 0.5f;
	  float y0 = y_center - h * 0.5f;	
	  obj.x_offset = x0;
	  obj.y_offset = y0;
	  obj.height = h;
	  obj.width = w;
	  obj.type = class_idx;
	  obj.score = box_prob;

	  objects.push_back(obj);
	}
      }  // class loop
    }    // point anchor loop
  }

  void
  YoloXP::qsortDescentInplace(ObjectArray & faceobjects, int left, int right)
  {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].score;

    while (i <= j) {
      while (faceobjects[i].score > p) {
	i++;
      }

      while (faceobjects[j].score < p) {
	j--;
      }

      if (i <= j) {
	// swap
	std::swap(faceobjects[i], faceobjects[j]);

	i++;
	j--;
      }
    }

#pragma omp parallel sections
    {
#pragma omp section
      {
	if (left < j) {
	  qsortDescentInplace(faceobjects, left, j);
	}
      }
#pragma omp section
      {
	if (i < right) {
	  qsortDescentInplace(faceobjects, i, right);
	}
      }
    }
  }

  void
  YoloXP::nmsSortedBboxes(
			  const ObjectArray & faceobjects, std::vector<int> & picked, float nms_threshold)
   {
    picked.clear();
    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
      cv::Rect rect(
		    faceobjects[i].x_offset, faceobjects[i].y_offset, faceobjects[i].width,
		    faceobjects[i].height);
      areas[i] = rect.area();
    }

    for (int i = 0; i < n; i++) {
      const Object & a = faceobjects[i];

      int keep = 1;
      for (int j = 0; j < static_cast<int>(picked.size()); j++) {
	const Object & b = faceobjects[picked[j]];

	// intersection over union
	float inter_area = intersectionArea(a, b);
	float union_area = areas[i] + areas[picked[j]] - inter_area;
	// float IoU = inter_area / union_area
	if (inter_area / union_area > nms_threshold) {
	  keep = 0;
	}
      }

      if (keep) {
	picked.push_back(i);
      }
    }
  }

  void
  YoloXP::decodeOutputs(const OutputVStream &output_stream, const InputVStream &input_stream, const float * prob, ObjectArray & objects, const cv::Size & img_size)
  {
    ObjectArray proposals;
    std::vector<GridAndStride> grid_strides;
    auto input_info = input_stream.get_info();
    const int input_width = input_info.shape.width;
    const int input_height = input_info.shape.height;
    const float scale = std::min(input_width / (float)img_size.width, input_height / (float)img_size.height);
    auto output_info = output_stream.get_info();
    hailo_quant_info_t quant_info = output_info.quant_info;
    generateGridsAndStride(input_width, input_height, m_output_strides, grid_strides);
    generateYoloXPProposals(grid_strides, prob, m_thresh, proposals, m_num_classes, quant_info);

    qsortDescentInplace(proposals);

    std::vector<int> picked;
    nmsSortedBboxes(proposals, picked, m_nms_thresh);

    int count = static_cast<int>(picked.size());
    objects.resize(count);
    float scale_x = input_width / (float)img_size.width;
    float scale_y = input_height / (float)img_size.height;  
    for (int i = 0; i < count; i++) {
      objects[i] = proposals[picked[i]];
      float x0, y0, x1, y1;
      // adjust offset to original unpadded
      if (scale == -1.0) {
	x0 = (objects[i].x_offset) / scale_x;
	y0 = (objects[i].y_offset) / scale_y;
	x1 = (objects[i].x_offset + objects[i].width) / scale_x;
	y1 = (objects[i].y_offset + objects[i].height) / scale_y;
      } else {
	x0 = (objects[i].x_offset) / scale;
	y0 = (objects[i].y_offset) / scale;
	x1 = (objects[i].x_offset + objects[i].width) / scale;
	y1 = (objects[i].y_offset + objects[i].height) / scale;
      }
      // clip
      x0 = clamp(x0, 0.f, static_cast<float>(img_size.width - 1));
      y0 = clamp(y0, 0.f, static_cast<float>(img_size.height - 1));
      x1 = clamp(x1, 0.f, static_cast<float>(img_size.width - 1));
      y1 = clamp(y1, 0.f, static_cast<float>(img_size.height - 1));

      objects[i].x_offset = x0;
      objects[i].y_offset = y0;
      objects[i].width = x1 - x0;
      objects[i].height = y1 - y0;
    }
  }

  void
  YoloXP::drawBBox(cv::Mat &img, ObjectArray &objects, const uint8_t *colormap)
  {
    char buff[128];
    for (const auto & object : objects) {
      const auto left = object.x_offset;
      const auto top = object.y_offset;
      const auto right = clamp(left + object.width, 0, img.cols);
      const auto bottom = clamp(top + object.height, 0, img.rows);
      const auto id = object.type;
      cv::rectangle(
		    img, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(colormap[3*id+2], colormap[3*id+1], colormap[3*id]), 2);
      sprintf(buff, "%2.0f%%", object.score * 100);
      cv::putText(img, buff, cv::Point(left, top), 0, 0.5, cv::Scalar(colormap[3*id+2], colormap[3*id+1], colormap[3*id]), 2);
    }
  }

  cv::Mat
  YoloXP::getMask(const uint8_t *prob, int classes, int width, int height, int out_w, int out_h, const unsigned char*colormap) {
    cv::Mat mask = cv::Mat::zeros(out_h, out_w, CV_8UC3);
    //NHWC
    // #pragma omp parallel for    
    for (int y = 0; y < out_h; y++) {
      for (int x = 0; x < out_w; x++) {
	uint8_t max = 0;
	int index = 0;
	for (int c = 0; c < classes; c++) {
	  //NHWC
	  uint8_t value = prob[c + classes * x + classes * width * y];
	  if (max < value) {
	    max = value;
	    index = c;

	  }
	}
	//argmax.at<unsigned char>(y, x) = index;
	mask.at<cv::Vec3b>(y, x)[0] = colormap[3 * index + 0];
	mask.at<cv::Vec3b>(y, x)[1] = colormap[3 * index + 1];
	mask.at<cv::Vec3b>(y, x)[2] = colormap[3 * index + 2];	
      }
    }
    return mask;
  }  
}
