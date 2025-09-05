# clientmanager/admin.py
from django.contrib import admin
from django.utils.text import Truncator
from .models import UserEvent

@admin.register(UserEvent)
class UserEventAdmin(admin.ModelAdmin):
    # What you see in the list view
    list_display = (
        "id",
        "timestamp",
        "username",
        "user_link",        # shows FK user id/username if present
        "event_type",
        "page",
        "element_short",
        "value_short",
        "ip_address",
    )
    ordering = ("-timestamp", "-id")
    date_hierarchy = "timestamp"
    list_per_page = 100

    # Right-side filters
    list_filter = ("event_type", "page", "username", "ip_address", "timestamp")

    # Top search bar
    search_fields = (
        "username",
        "page",
        "element",
        "value",
        "ip_address",
        "user__username",
        "user__id",
    )

    # Performance + safety
    list_select_related = ("user",)
    raw_id_fields = ("user",)  # faster user FK picker
    readonly_fields = (
        "user",
        "username",
        "ip_address",
        "page",
        "event_type",
        "element",
        "value",
        "timestamp",
    )

    # -------- Display helpers --------
    def user_link(self, obj):
        if obj.user_id and getattr(obj.user, "username", None):
            return f"{obj.user_id} / {obj.user.username}"
        return "—"
    user_link.short_description = "User"

    def element_short(self, obj):
        return Truncator(obj.element or "").chars(60)
    element_short.short_description = "element"

    def value_short(self, obj):
        return Truncator(obj.value or "").chars(60)
    value_short.short_description = "value"