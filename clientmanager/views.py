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
    return render(request, 'clientmanager/calendar2.html')
    
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
    
