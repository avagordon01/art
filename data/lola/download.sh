#!/usr/bin/env bash

set -o errexit

#download all text and metadata
#lftp is much faster than wget because it is parallel and there are many small files
#takes about half an hour, downloads abour 4GB
# lftp -c mirror \
#     --continue --only-missing --parallel=64 --use-cache \
#     -I *.htm -I *.html -I *.cat -I *.fmt -I *.txt -I *.lbl -I *.xml \
#     https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/
# exit

#see lro_mission.txt for explanation of different mission phases
#commissioning, nominal, science mission, extended science
#lro_{phase}_{orbit}
#lro_co
#lro_no_01 - lro_no_13
#lro_sm_01 - lro_sm_26
#lro_es_01 - lro_es_136

#download binary data
lftp -c mirror \
    --continue --only-missing --parallel=64 --use-cache \
    -I 'data/lola_rdr/lro_es_110/*.dat' \
    https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/
