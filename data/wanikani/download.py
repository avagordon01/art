#!/usr/bin/env -S python -i

import pickle
import datetime
import json
from pprint import pprint as print

from wanikani_api.client import Client
with open('secrets.json') as f:
    v2_api_key = json.load(f)['v2_api_key']
client = Client(v2_api_key)

time_format = "%Y-%m-%d %H:%M:%S"
dt = datetime.datetime.strptime('2021-07-26 17:01:27', time_format)

user_information = client.user_information()

def keep_keys(dict, keys):
    return {key: dict[key] for key in keys if key in dict}

reviews_keys = ['subject', 'ending_srs_stage', 'starting_srs_stage', 'id', 'assignment_id', 'subject_id', 'data_updated_at']
raw_reviews = [keep_keys(vars(x), reviews_keys) for x in client.reviews(fetch_all=True)]
#reviews = {x.id: x for x in raw_reviews}

assignments_keys = ['subject_type', 'srs_stage', 'id', 'subject_id', 'available_at', 'burned_at', 'created_at', 'passed_at', 'resurrected_at_at', 'started_at', 'unlocked_at']
raw_assignments = [keep_keys(vars(x), assignments_keys) for x in client.assignments(fetch_all=True)]
#assignments = {x.id: x for x in raw_assignments}

subjects_keys = ['resource', 'auxiliary_meanings', 'characters', 'meanings', 'readings', 'level', 'id', 'component_subeject_ids', 'amalgamation_subject_ids', 'created_at', 'data_updated_at']
raw_subjects = [keep_keys(vars(x), subjects_keys) for x in client.subjects(fetch_all=True)]
#subjects = {x.id: x for x in raw_subjects}

data = {
    'user_information': user_information,
    'raw_reviews': raw_reviews,
    'raw_assignments': raw_assignments,
    'raw_subjects': raw_subjects,
}

with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
