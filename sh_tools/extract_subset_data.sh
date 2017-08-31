#!/bin/bash

for target in train dev; do
    cat feature_scp/${target}.scp | sed -e '1~3d' >  ${target}.scp
    cat bounds/phn/${target}.phn | sed -e '1~3d' >  ${target}.phn
done
