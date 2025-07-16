from django.urls import path
from . import views 

urlpatterns = [
    path('', views.home_view, name='home'),
    path('404/', views.error_404_view, name='404'),
    path('blank/', views.blank_view, name='blank'),
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    path('login/', views.login_view, name='login'),
    path('profile/', views.profile_view, name='profile'),
    path('register/', views.register_view, name='register'),
    path('table/', views.table_view, name='table'),
    path('calendar/', views.calendar_view, name='calendar'),
    path('calendar2/', views.calendar2_view, name='calendar2'),
    path('client-engagement-dashboard/', views.client_engagement_dashboard_view, name='client_engagement_dashboard'),  
    path('project-management-form/', views.project_managment_page_view, name='project_management_page'),  
    path('reporting-page/', views.reporting_page_view, name='reporting_page'),  
]