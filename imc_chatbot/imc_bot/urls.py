from django.urls import path
from . import views

urlpatterns = [
    path('', views.your_view, name='home'),
    # add other url patterns here
]
