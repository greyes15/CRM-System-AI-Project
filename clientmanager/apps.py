from django.apps import AppConfig


class ClientmanagerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "clientmanager"
    
    def ready(self):
        # This code runs once, at Django startup
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")