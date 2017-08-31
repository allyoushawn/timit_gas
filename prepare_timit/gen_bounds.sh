#!/bin/bash

rm -rf ../bounds/phn
mkdir -p ../bounds/phn

for target in train test; do

    while read -ra seg; do
        bound_file=${seg[4]/.wav/.phn}
        echo "${seg[0]} ${bound_file}" >>../bounds/phn/${target}.phn

    done < ${target}.wav.scp

    phn_file="../bounds/phn/${target}.phn"
    #sort first
    sort -k1,1V $phn_file >"${target}.phn"
    feature_scp="../feature_scp/${target}.scp"


    while read feature_line ; do

        read -ra feature_line_seg <<<$feature_line
        read -u 3  bound_line
        read -ra bound_line_seg <<<$bound_line

        while [ "${feature_line_seg[0]}" != "${bound_line_seg[0]}" ]; do
            read -u 3  bound_line
            read -ra bound_line_seg <<<$bound_line
        done

        echo $bound_line >>"ali_${target}.phn"

    done <$feature_scp 3<"${target}.phn"

    mv "ali_${target}.phn" "../bounds/phn/${target}.phn"
    rm -f "${target}.phn"

done
