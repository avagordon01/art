#!/usr/bin/env -S python -i

import datetime
import json

from wanikani_api.client import Client
with open('secrets.json') as f:
    v2_api_key = json.load(f)['v2_api_key']
client = Client(v2_api_key)
#user_information = client.user_information()
#print(user_information)
if False:
    vocabulary = client.subjects(types=["vocabulary"], fetch_all=False)
    for vocab in vocabulary:
        print(vocab)
        #if len(vocab.readings) > 1 or len(vocab.meanings) > 1:
        #print(f"{[x.reading for x in vocab.readings]} is {[x.meaning for x in vocab.meanings]}")


time_format = "%Y-%m-%d %H:%M:%S"
dt = datetime.datetime.strptime('2021-07-26 17:01:27', time_format)
if False:
    reviews = client.reviews(fetch_all=False, updated_after=dt)
    for review in reviews:
        print(review.id, review.srs_stage, review.resource, review.subject, review.subject_id, review.created_at, review.data_updated_at)

assignments = client.assignments(fetch_all=False, updated_after=dt)
for assignment in assignments:
    print(assignment.id, assignment.srs_stage, assignment.subject, assignment.subject_id, assignment.subject_type)
