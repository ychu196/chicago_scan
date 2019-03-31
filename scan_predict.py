# Image classification using AWS Sagemaker and Linear Learner
# Program set up and import libraries
import numpy as np
import pandas as pd
import os

from sagemaker import get_execution_role

role = get_execution_role()
bucket = 'chi-hackathon-skin-images'

# Import Data
import boto3
from sagemaker import get_execution_role

role = get_execution_role()
bucket='chi-hackathon-skin-images'
data_key = 'ISIC_0000000.json' # need a way to go through entire library
data_location = 's3://{}/{}'.format(bucket, data_key)

metadata_set = pd.read_json(data_location)
image_set = np.asarray(data_location)
# TBD - transform json data to array
# TBD - transform image data to dataframe
train_set = zip(image_set, metadata_set)


# Split Data into Train and Validate
import random
random.seed(9001)
split = np.random.rand(len(df)) < 0.8
valid_set = train_set[split]
train_set = train_set[~split]

# Train Model
import boto
import sagemaker

data_location = 's3://{}/linearlearner_highlevel_example/data'.format(bucket)
output_location = 's3://{}/linearlearner_highlevel_example/output'.format(bucket)
print('training data will be uploaded to: {}'.format(data_location))
print('training artifacts will be uploaded to: {}'.format(output_location))

sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(container, role, train_instance_count=1, rain_instance_type='ml.c4.xlarge',
    output_path=output_location, sagemaker_session=sess)
linear.set_hyperparameters(feature_dim=784, predictor_type='binary_classifier', mini_batch_size=200)

linear.fit({'train': train_set})

# Deploy Model
linear_predictor = linear.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

# Validate
from sagemaker.predictor import csv_serializer, json_deserializer

linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer

result = linear_predictor.predict(train_set[0][30:31])
print(result)
