#!/bin/bash
patt='business_id\"\: \"'
while IFS='' read -r line || [[ -n "$line" ]]; do
    grep -w $patt+$line yelp_academic_dataset_review_vegas.json 
done < "$1"
