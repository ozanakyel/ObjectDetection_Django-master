from django.urls import path

from . import views

print('sdfisfşs')

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
]