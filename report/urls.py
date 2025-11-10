from . import views
from django.urls import path

app_name = 'report'

urlpatterns = [
    path('exportar_reporte/', views.exportar_reporte, name='exportar_reporte'),

    path('reporte_distribucion_frecuencias/<str:file_name>', views.reporte_distribucion_frecuencias, name='reporte_distribucion_frecuencias'),
    path('reporte_tendencia_central/<str:file_name>', views.reporte_tendencia_central, name='reporte_tendencia_central'),
    path('reporte_variabilidad/<str:file_name>', views.reporte_variabilidad, name='reporte_variabilidad'),    


    path('reporte_regresion_linear/<str:file_name>', views.reporte_regresion_linear, name='reporte_regresion_linear'),
    path('reporte_regresion_logistica/<str:file_name>', views.reporte_regresion_logistica, name='reporte_regresion_logistica'),  
    path('reporte_correlacion_pearson/<str:file_name>', views.reporte_correlacion_pearson, name='reporte_correlacion_pearson'),
    path('reporte_correlacion_spearman/<str:file_name>', views.reporte_correlacion_spearman, name='reporte_correlacion_spearman'),  
    path('reporte_kmeans_clustering/<str:file_name>', views.reporte_kmeans_clustering, name='reporte_kmeans_clustering'),  
    path('reporte_arbol_decision_clasificacion/<str:file_name>', views.reporte_arbol_decision_clasificacion, name='reporte_arbol_decision_clasificacion'),


    path('reporte_intervalo_confianza/<str:file_name>', views.reporte_intervalo_confianza, name='reporte_intervalo_confianza'),  
    path('reporte_prueba_hipotesis/<str:file_name>', views.reporte_prueba_hipotesis, name='reporte_prueba_hipotesis'),
    path('reporte_perceptron/<str:file_name>', views.reporte_perceptron, name='reporte_perceptron'),    

]
