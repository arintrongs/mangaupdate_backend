# api/serializers.py

from rest_framework import serializers
from .models import predict

class PredictSerializer(serializers.ModelSerializer):

    class Meta:
        model = predict
        fields = ('genres','steps')