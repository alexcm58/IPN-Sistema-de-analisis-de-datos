from django.urls import path
from . import views

app_name = 'file_handler'

urlpatterns = [
    path('', views.cargar_archivo, name='cargar_archivo'),
    path('revisar_csv/<str:file_name>/', views.revisar_csv, name='revisar_csv'),
    path('cargar_mas_filas/<str:file_name>/', views.cargar_mas_filas, name='cargar_mas_filas'),
    path('descargar_csv/<str:file_name>/', views.descargar_archivo_csv, name='descargar_csv'),
    # path('eliminar_csv/<str:file_name>/', views.eliminar_csv, name='eliminar_csv'),
    path('cambiar_version/<str:file_name>/<str:direction>/', views.cambiar_version, name='cambiar_version')
]

