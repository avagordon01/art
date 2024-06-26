#!/usr/bin/env bash

set -o errexit

#download all text and metadata
wget --recursive --level=inf --no-parent --continue -e robots=off --ignore-case \
    --accept htm,html,cat,fmt,txt,lbl,xml \
    https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/

exit

#see lro_mission.txt for explanation of different mission phases
#commissioning, nominal, science mission, extended science
#lro_{phase}_{orbit}
#lro_co
#lro_no_01 - lro_no_13
#lro_sm_01 - lro_sm_26
#lro_es_01 - lro_es_136

#download binary data
wget --recursive --level=inf --no-parent --continue -e robots=off \
    https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_es_110/
    #https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_co/ \
    #https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_no_01/ \
    #https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_no_13/ \
    #https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_sm_01/ \
    #https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_sm_26/ \
    #https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_es_01/ \
    #https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/lro_es_118/
