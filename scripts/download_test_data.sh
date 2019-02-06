#!/bin/bash

MODULES=(oap2dt3d oap2dt3dFuncTests oapDeviceTests oapNeural)

for module in ${MODULES[*]}
do
  bash ./download_module_data.sh $module $1
done
