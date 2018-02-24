# api/models.py

from django.db import models

class predict(models.Model):
    genres = models.CharField(max_length=255)
    steps = models.IntegerField()


