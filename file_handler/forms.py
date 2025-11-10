import csv
import os
from django import forms

# Formulario para cargar un archivo CSV, 
# aquí se define el campo archivo_csv que es un FileField y se limita a archivos con extensión .csv 
# tambien como se define un mensaje de error en caso de que el archivo no sea un CSV.

class CargaCSVForm(forms.Form):
    archivo_csv = forms.FileField(
        widget=forms.FileInput(attrs={'accept': '.csv', 'class': 'form-control', 'id': 'formFile'}),
        label=''
    )

    def clean_archivo_csv(self):
        file = self.cleaned_data['archivo_csv']
        extension = os.path.splitext(file.name)[1]
        if not extension.lower() == ".csv":
            raise forms.ValidationError('Solo se admiten archivos CSV.')
        return file
    










    