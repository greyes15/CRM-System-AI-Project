# clientmanager/admin.py
import csv
from django.contrib import admin
from django.http import HttpResponse
from django.utils.text import Truncator
from .models import UserEvent, ChatMessage


@admin.action(description="Export selected records to CSV")
def export_as_csv(modeladmin, request, queryset):
    meta = modeladmin.model._meta
    field_names = [field.name for field in meta.fields]

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{meta.model_name}.csv"'

    writer = csv.writer(response)
    writer.writerow(field_names)

    for obj in queryset:
        writer.writerow([getattr(obj, field) for field in field_names])

    return response

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
    actions = [export_as_csv]

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


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("id", "timestamp", "username", "message_type", "page", "ip_address", "content_short")
    ordering = ("-timestamp", "-id")
    date_hierarchy = "timestamp"
    list_per_page = 100
    actions = [export_as_csv]

    list_filter = ("message_type", "page", "username", "ip_address", "timestamp")
    search_fields = ("username", "content", "ip_address", "user__username")

    list_select_related = ("user",)
    raw_id_fields = ("user",)
    readonly_fields = ("user", "username", "ip_address", "page", "message_type", "content", "timestamp")

    def content_short(self, obj):
        return Truncator(obj.content or "").chars(80)
    content_short.short_description = "content"