#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:50:30 2023

@author: nicholas.lusk
"""

import io
import os
import sys
import csv
import boto3

import pandas as pd
import dask.array as da

"""Accessing the S3 buckets using boto3 client"""
s3_client =boto3.client('s3')
s3_bucket_name='aind-open-data'
s3 = boto3.resource('s3',
                    aws_access_key_id= 'AKIA543K5JHJSPMKKG7F',
                    aws_secret_access_key='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')

""" Getting data files from the AWS S3 bucket"""
my_bucket=s3.Bucket(s3_bucket_name)
pref = '/SmartSPIM_644106_2022-12-09_12-12-39_stitched_2022-12-16_16-55-11/processed/OMEZarr'
bucket_list = []
for obj in my_bucket.objects.filter(Prefix = pref):
    print(obj.key)
   #if not os.path.exists(os.path.dirname(obj.key)):
   #         os.makedirs(os.path.dirname(obj.key))
   #     bucket.download_file(obj.key, obj.key)
