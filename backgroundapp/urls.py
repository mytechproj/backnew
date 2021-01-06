from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from backgroundapp import views

urlpatterns = [   
    path('',views.home_view,name="home"),
    path('about',views.about,name="about"),
    path('services',views.services,name="services"),
    path('contact',views.contact,name="contact"),
    path('login',views.loginuser,name="loginuser"),
    path('logout',views.logoutuser,name="logoutuser"),
]

urlpatterns+= static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.MYMEDIA_URL, document_root=settings.MYMEDIA_ROOT)
    