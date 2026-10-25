from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/health", views.api_health),
    path("api/documents", views.api_documents),
    path("api/documents/<str:doc_id>", views.api_document),
    path("api/documents/<str:doc_id>/page/<int:page>.png", views.api_page_image),
    path("api/ask", views.api_ask),
    path("api/history", views.api_history),
    path("api/samples", views.api_samples),
    path("api/evaluate", views.api_evaluate),
]
