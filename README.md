# HailoRT YOLOXP for Low Power Edge AI


https://github.com/tier4/hailort-yoloXP/assets/127905436/0cdbc38f-f980-4bc8-aea0-2b845f876698


## Purpose

This package implements YOLOXP on Hailo-8 chip with HailoRT for low power inference.
## Setup

Install HailoRT from Hailo (https://hailo.ai/ja/developer-zone/)

Moreover, you need to install as bellow.
```bash
sudo apt-get install libgflags-dev libboost-all-dev
```

## Building

```bash
git clone git@github.com:tier4/hailort-yoloXP.git
cd hailort-yoloXP
cd build/
cmake ..
make -j
```

## Start inference

-Infer from a Video

```bash
/hailort-yoloXP --hef {HEF Name} --v {Video Name}  --thresh {Score Thresh} --c {numClasses}
```

### Cite

Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun, "YOLOX: Exceeding YOLO Series in 2021", arXiv preprint arXiv:2107.08430, 2021 [[ref](https://arxiv.org/abs/2107.08430)]


## Parameters

--hef=HEF Path (Hailo Model)

--v=Video Path

--d=Directory Path

--c=Number of Classes

--thresh=Score Thresh

--nmsThresh=NMS Thresh

## Assumptions / Known limits

### Todo

- [] Support Multi-Precision execution
- [] Optimize Postprocess
- [] NMS and Argmax on chip processing
- [] Support Hailo-15

## Reference repositories

- <https://github.com/Megvii-BaseDetection/YOLOX>
- <https://github.com/hailo-ai/Hailo-Application-Code-Examples>
