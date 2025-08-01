from django.urls import path
from . import views 

urlpatterns = [
    path('', views.home_view, name='home'),
    path('404/', views.error_404_view, name='404'),
    path('blank/', views.blank_view, name='blank'),
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    path('login/', views.login_view, name='login'),
    path('profile/', views.profile_view, name='profile'),
    path('profile-main/', views.profile_main_view, name='profile_main'),
    path('register/', views.register_view, name='register'),
    path('table/', views.table_view, name='table'),
    path('table-main/', views.table_main_view, name='table_main'),
    path('calendar/', views.calendar_view, name='calendar'),
    path('calendar2/', views.calendar2_view, name='calendar2'),
    path('calendar2-main/', views.calendar2_main_view, name='calendar2_main'),
    path('client-engagement-dashboard/', views.client_engagement_dashboard_view, name='client_engagement_dashboard'),  
    path('project-management-form/', views.project_managment_page_view, name='project_management_page'),  
    path('reporting-page/', views.reporting_page_view, name='reporting_page'),  
    path('team-task-main/', views.team_task_main_view, name='team_task_main'),  
    path('team-task/', views.team_task_view, name='team_task'),  
    path('AIChatBox/', views.AIChatBox_view, name='AIChatBox'),  
    path("chat-proxy/", views.chat_proxy, name="chat_proxy"),
]