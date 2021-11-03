#!/bin/bash
set -euxo pipefail 

mkdir -p ./data/phys101
cd ./data/phys101
curl -O http://phys101.csail.mit.edu/data/phys101_v1.0.zip
unzip -q phys101_v1.0.zip
rm phys101_v1.0.zip
