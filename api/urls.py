# api/urls.py

from django.conf.urls import url, include
from rest_framework.urlpatterns import format_suffix_patterns

from .views import request_predict

urlpatterns = {
    url(r'^predict$', request_predict),
}

urlpatterns = format_suffix_patterns(urlpatterns)
