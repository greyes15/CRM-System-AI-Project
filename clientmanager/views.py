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
    
def reporting_tool_view(request):
    return render(request, 'clientmanager/reporting_tool.html')
    
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
    
@xframe_options_sameorigin
def reporting_tool_main_view(request):
    return render(request, 'clientmanager/reporting_tool_main.html')

    
# Replace the chat_proxy section in your views.py with this:

import os, json
from openai import OpenAI
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from .rag_engine import retrieve_top_k, refresh_cache

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

RAG_TOP_K = 6
RAG_MIN_SCORE = 0.45  # tweak as needed for your manual
RAG_PREAMBLE = (
    "Answer using ONLY the excerpts from the EMPLOYEE USE MANUAL provided below. "
    "Cite the page numbers (e.g., 'p. 17'). "
    "If the excerpts do not contain the answer, say you don't have enough information and suggest contacting the Graduate Center for Excellence."
)

def _latest_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"]).strip()
    return ""

@csrf_exempt
@require_POST
def chat_proxy(request):
    try:
        payload  = json.loads(request.body)
        messages = payload.get("messages", [])

        user_q = _latest_user_text(messages)
        hits = retrieve_top_k(user_q, k=RAG_TOP_K) if user_q else []

        if hits:
            # Filter very weak matches; ensure at least some context remains
            good = [h for h in hits if h["score"] >= RAG_MIN_SCORE] or hits[:3]

            # Build a compact context block with page cites
            lines = []
            for i, h in enumerate(good, start=1):
                lines.append(f"{i}) (p. {h['page']}) {h['text']}")
            ctx = f"{RAG_PREAMBLE}\n\nExcerpts:\n" + "\n\n".join(lines)

            # Prepend as a system message
            messages = [{"role": "system", "content": ctx}] + messages
        else:
            # Hard fallback: ask model to be honest about missing context
            messages = [{"role": "system", "content": "No manual excerpts available; do not guess."}] + messages

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return JsonResponse(completion.model_dump(), safe=False)

    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)

@require_GET
def rag_refresh(request):
    try:
        n = refresh_cache()
        return JsonResponse({"ok": True, "chunks": n})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
        
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import json
from .models import UserEvent
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import json
from .models import UserEvent

def get_client_ip(request):
    """Helper to extract IP address from request headers."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

# Add this to your views.py

from django.views.decorators.http import require_GET
from django.http import JsonResponse

@require_GET
def test_rag(request):
    """Test the RAG engine step by step."""
    try:
        from .rag_engine import health_check, retrieve_top_k, get_index_info
        
        # Run health check
        health = health_check()
        
        if health["status"] == "error":
            return JsonResponse({
                "success": False,
                "step": "health_check",
                "health": health
            })
        
        # Test retrieval with a simple query
        query = request.GET.get('q', 'vacation policy')
        results = retrieve_top_k(query, k=3)
        
        return JsonResponse({
            "success": True,
            "health": health,
            "query": query,
            "results_count": len(results),
            "results": results[:2],  # Show first 2 results
            "index_info": get_index_info()
        })
        
    except Exception as e:
        import traceback
        return JsonResponse({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })

# Also update your chat_proxy to be simpler and more robust
@csrf_exempt
@require_POST
def chat_proxy(request):
    """Handle chat requests with RAG."""
    try:
        # Parse request
        payload = json.loads(request.body)
        messages = payload.get("messages", [])
        
        # Get latest user message
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "").strip()
                break
        
        # If we have a query, try RAG retrieval
        context_message = None
        if user_query:
            from .rag_engine import retrieve_top_k
            
            try:
                # Get relevant chunks
                chunks = retrieve_top_k(user_query, k=5)
                
                if chunks:
                    # Filter chunks with decent scores
                    good_chunks = [c for c in chunks if c["score"] > 0.3]
                    if not good_chunks:
                        good_chunks = chunks[:3]  # Use top 3 if none are good enough
                    
                    # Build context
                    context_parts = [
                        "Answer based on the Employee Manual excerpts below. Cite page numbers when possible.",
                        "If the excerpts don't contain enough information, say so clearly.",
                        "",
                        "Relevant excerpts:"
                    ]
                    
                    for i, chunk in enumerate(good_chunks, 1):
                        context_parts.append(f"{i}. (Page {chunk['page']}) {chunk['text']}")
                    
                    context_message = {
                        "role": "system", 
                        "content": "\n".join(context_parts)
                    }
                
            except Exception as rag_error:
                logger.error(f"RAG error: {rag_error}")
                # Continue without RAG context
        
        # Build final messages
        final_messages = []
        if context_message:
            final_messages.append(context_message)
        final_messages.extend(messages)
        
        # Call OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=final_messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return JsonResponse(completion.model_dump(), safe=False)
        
    except Exception as e:
        logger.error(f"Chat proxy error: {e}")
        return JsonResponse({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'm having trouble processing your request. Please try again."
                }
            }]
        }, status=500)

# views.py
import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib.auth.models import User
# from .models import UserEvent
# from .utils import get_client_ip  # wherever your helper lives

@require_POST
def track_event(request):
    # 1) Parse body (JSON or form fallback)
    try:
        body = request.body.decode("utf-8") if isinstance(request.body, (bytes, bytearray)) else request.body
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        data = request.POST.dict()

    # 2) Pull fields with light sanitization
    page = (data.get("page") or "")[:512]
    event_type = (data.get("event_type") or "")[:64]
    element = (data.get("element") or "")[:512]
    value = data.get("value") or ""

    # 3) Get participant ID (your “username”) -> payload > cookie > auth user > 'guest'
    username = (
        (data.get("username") or "").strip()
        or (request.COOKIES.get("username") or "").strip()
        or (request.user.username if request.user.is_authenticated else "")
        or "guest"
    )
    username = username[:150]  # Django's default User.username max_length is 150

    # 4) Ensure we have a User to satisfy the FK (no auth required)
    user_obj, created = User.objects.get_or_create(
        username=username,
        defaults={"email": f"{username}@research.invalid"}
    )
    if created:
        user_obj.set_unusable_password()
        user_obj.save(update_fields=["password"])

    # 5) Save the event
    UserEvent.objects.create(
        user=user_obj,
        username=username,
        ip_address=get_client_ip(request),
        page=page,
        event_type=event_type,
        element=element,
        value=value,
    )

    return JsonResponse({"status": "ok"})
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")  # must match <input name="">
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("index")  # replace with your dashboard/home
        else:
            messages.error(request, "Invalid username or password")
    return render(request, "clientmanager/login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if password1 != password2:
            messages.error(request, "Passwords do not match")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken")
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered")
        else:
            user = User.objects.create_user(username=username, email=email, password=password1)
            login(request, user)  # auto login
            return redirect("index")

    return render(request, "clientmanager/register.html")
    
# views.py
from pathlib import Path
from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.clickjacking import xframe_options_sameorigin

# Where your PDFs live. Put files in:  <project>/media/pdfs/...
PDF_DIR = (Path(getattr(settings, "MEDIA_ROOT", Path(settings.BASE_DIR) / "media")) / "pdfs").resolve()


@xframe_options_sameorigin
def employee_user_manual(request):
    """
    Renders a page that shows the Employee User Manual PDF in an iframe.
    The template will reference the PDF via {% static %}.
    """
    return render(request, "clientmanager/employee_user_manual.html")