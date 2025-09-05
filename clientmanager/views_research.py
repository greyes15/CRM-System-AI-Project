# clientmanager/views_research.py
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

@require_http_methods(["GET", "POST"])
def research_login_view(request):
    if request.method == "POST":
        pid = (request.POST.get("participant_id") or "").strip()
        if not pid:
            return render(request, "clientmanager/research_login.html", {"error": "Participant ID is required."})
        # Save as “username” cookie (30 days)
        resp = redirect("home")
        resp.set_cookie("username", pid, max_age=60*60*24*30, samesite="Lax")
        return resp
    return render(request, "clientmanager/research_login.html")
    