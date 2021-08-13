#!/usr/bin/env -S python -i

import pickle
import json
from pprint import pprint as print

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

user_information = data['user_information']
raw_reviews = data['raw_reviews']
raw_assignments = data['raw_assignments']
raw_subjects = data['raw_subjects']

dict_reviews = {x['id']: x for x in raw_reviews}
dict_assignments = {x['id']: x for x in raw_assignments}
dict_subjects = {x['id']: x for x in raw_subjects}

reviews = raw_reviews
for r in reviews:
    del r['assignment_id']
    del r['id']
    del r['subject_id']
    del r['starting_srs_stage']

reviews.sort(key=lambda x: x['data_updated_at'])

output = reviews
for x in output:
    x['data_updated_at'] = x['data_updated_at'].timestamp()

output = list(output)

timestamps = sorted([x['data_updated_at'] for x in output])
start_timestamp = timestamps[0]
end_timestamp = timestamps[-1]
output = {'reviews': output, 'start_timestamp': start_timestamp, 'end_timestamp': end_timestamp}

with open('data2.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
