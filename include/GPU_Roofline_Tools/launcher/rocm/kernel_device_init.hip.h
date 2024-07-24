#pragma once

void initHIPDevice();
void setHIPDevice(uint32_t device_index);
uint32_t getMaxWorkgroupSize(uint32_t device_index);
uint32_t getWaveFrontSize(uint32_t device_index);