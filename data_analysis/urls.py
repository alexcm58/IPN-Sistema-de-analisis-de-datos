from . import views
from django.urls import path

app_name = 'data_analysis'

# URLs para las vistas de análisis de datos

urlpatterns = [
   path('distribucion_frecuencias/<str:file_name>', views.distribucion_frecuencias, name='distribucion_frecuencias'),
   path('tendencia_central/<str:file_name>', views.tendencia_central, name='tendencia_central'),
   path('variabilidad/<str:file_name>', views.variabilidad, name='variabilidad'),

   path('regresion_lineal/<str:file_name>', views.regresion_lineal, name='regresion_lineal'),
   path('regresion_logistica/<str:file_name>', views.regresion_logistica, name='regresion_logistica'),
   path('correlacion_pearson/<str:file_name>', views.correlacion_pearson, name='correlacion_pearson'),
   path('correlacion_spearman/<str:file_name>', views.correlacion_spearman, name='correlacion_spearman'),
   path('kmeans_clustering/<str:file_name>', views.kmeans_clustering, name='kmeans_clustering'),
   path('arbol_decision_clasificacion/<str:file_name>', views.arbol_decision_clasificacion, name='arbol_decision_clasificacion'),

   path('intervalo/<str:file_name>', views.intervalo, name='intervalo'),
   path('prueba_hipotesis/<str:file_name>', views.prueba_hipotesis, name='prueba_hipotesis'),
   path('perceptron/<str:file_name>', views.perceptron, name='perceptron'),


]
