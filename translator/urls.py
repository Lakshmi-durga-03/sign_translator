from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('process_frame/', views.process_frame, name='process_frame'),
    path('clear_text/', views.clear_text, name='clear_text'),
    path('update_text/', views.update_text, name='update_text'),
    path('text_to_speech/', views.text_to_speech, name='text_to_speech'),
    path('speech_to_text/', views.speech_to_text, name='speech_to_text'),
    path('train_model/', views.train_model, name='train_model'),
]


