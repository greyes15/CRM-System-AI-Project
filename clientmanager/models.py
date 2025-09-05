from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class UserEvent(models.Model):
    EVENT_TYPES = [
        ("click", "Click"),
        ("input", "Input"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    username = models.CharField(max_length=150, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    page = models.CharField(max_length=255)
    event_type = models.CharField(max_length=10, choices=EVENT_TYPES)
    element = models.TextField()
    value = models.TextField(blank=True, null=True)  # for keyboard input
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        # show newest events first in admin
        ordering = ("-timestamp", "-id")

    def __str__(self):
        return f"{self.username} ({self.ip_address}) {self.event_type} on {self.page}"
