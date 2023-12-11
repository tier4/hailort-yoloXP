#include "colormap.hpp"

const unsigned char colormap[] = {
  255,255,255, //UNKNOWN
  0,0,255, //CAR
  0,160,165, //TRUCK
  100,0,200, //BUS
  128,255,0, //BICYCLE
  255,255,0, //MOTORBIKE
  255,0,32, //PEDESTRIAN
  255,0,0, //ANIMAL  
};

const unsigned char semseg_colormap[] = {
  0,0,0,
  70,70,70,
  255,0,255,
  30,170,250,
  0,220,220,
  60,20,220,
  142,0,0,
  230,0,0,
  128,64,128,
  128,128,128,
  194,253,147,
  255,206,135,
};
