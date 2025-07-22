from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_sameorigin

# Create your views here.
def home_view(request):
    return render(request, 'clientmanager/index.html')
    
def error_404_view(request):
    return render(request, 'clientmanager/404.html')

def blank_view(request):
    return render(request, 'clientmanager/blank.html')

def forgot_password_view(request):
    return render(request, 'clientmanager/forgot-password.html')

def login_view(request):
    return render(request, 'clientmanager/login.html')

def profile_view(request):
    return render(request, 'clientmanager/profile.html')

def register_view(request):
    return render(request, 'clientmanager/register.html')

def table_view(request):
    return render(request, 'clientmanager/table.html')

def calendar_view(request):
    return render(request, 'clientmanager/calendar.html')
    
def calendar2_view(request):
    return render(request, 'clientmanager/Calendar2.html')
    
def team_task_view(request):
    return render(request, 'clientmanager/team_task.html')
    
#def client_engagement_dashboard_view(request):
#    return render(request, 'clientmanager/client_engagement_dashboard.html')
    
@xframe_options_sameorigin
def client_engagement_dashboard_view(request):
    return render(request, 'clientmanager/client_engagement_dashboard.html')

@xframe_options_sameorigin
def reporting_page_view(request):
    return render(request, 'clientmanager/Reporting_page.html')

@xframe_options_sameorigin
def project_managment_page_view(request):
    return render(request, 'clientmanager/Project_management_page.html')
    
@xframe_options_sameorigin
def calendar2_main_view(request):
    return render(request, 'clientmanager/calendar2_main.html')
    
@xframe_options_sameorigin
def table_main_view(request):
    return render(request, 'clientmanager/table_main.html')
    
@xframe_options_sameorigin
def profile_main_view(request):
    return render(request, 'clientmanager/profile_main.html')
    
@xframe_options_sameorigin
def team_task_main_view(request):
    return render(request, 'clientmanager/team_task_main.html')
    
@xframe_options_sameorigin
def AIChatBox_view(request):
    return render(request, 'clientmanager/AIChatBox.html')
    
import os
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json


def chat_api(request):
    api_key = os.getenv("OPENAI_API_KEY")  # Get from env
    return render(request, "chat.html", {"openai_api_key": api_key})
