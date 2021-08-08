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

dates = \
    [x[y] for x in dict_reviews.values() for y in ['data_updated_at']] + \
    [x[y] for x in dict_assignments.values() for y in ['burned_at', 'passed_at', 'started_at', 'unlocked_at']] + \
    [x[y] for x in dict_subjects.values() for y in ['data_updated_at']]

for k, v in dict_assignments.items():
    date_labels = ['unlocked_at', 'created_at', 'started_at', 'passed_at', 'burned_at', 'available_at']
    dates = [v[x] for x in date_labels if x in v and v[x]]
    assert dates == sorted(dates)

print(len(dict_reviews))

reviews_by_subject = {}
for r in dict_reviews.values():
    subject = r['subject_id']
    del r['assignment_id']
    del r['id']
    del r['subject_id']
    del r['starting_srs_stage']
    if subject in reviews_by_subject:
        reviews_by_subject[subject]['updates'].append(r)
    else:
        reviews_by_subject[subject] = {'updates': [r]}

for id, s in dict_subjects.items():
    if id not in reviews_by_subject:
        continue
    if not s['characters']:
        continue
    if len(s['characters']) == 1:
        reviews_by_subject[id]['character'] = s['characters']
    else:
        del reviews_by_subject[id]

for r in reviews_by_subject.values():
    r['updates'].sort(key=lambda x: x['data_updated_at'])

print(reviews_by_subject)
