// Este script incluye las funciones necesarias para exportar un reporte en formato .docx
// Los datos se obtienen de las tablas y gráficas presentes en la página actual, es decir la pagina del reporte correspondiente.
// Una vez son extraidos los datos y las imagenes, se crea un archivo .docx con los datos obtenidos y se descarga en el navegador.
// Las funciones incluidas son: obtenerCSRFToken, convertirTablaAJSON, 
// convertirGraficasAImagenesBase64, exportarReporte y enviarReporteAlServidor

function obtenerCSRFToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]').value;
}

function convertirTablaAJSON(selector) {
    const todasLasTablas = document.querySelectorAll(selector);
    const dataDeTodasLasTablas = Array.from(todasLasTablas).map(tabla => {
        let tituloElemento = tabla.closest('.table-responsive').previousElementSibling;
        
        const titulo = tituloElemento && tituloElemento.tagName === 'H4' ? tituloElemento.innerText : 'Sin título';
        
        const filas = Array.from(tabla.rows).map(tr => {
            return Array.from(tr.cells).map(td => td.textContent.trim());
        });
        return { titulo, filas };
    });
    return dataDeTodasLasTablas;
}

function convertirGraficasAImagenesBase64() {
    const graficos = document.querySelectorAll('canvas');
    const graficosIds = Array.from(graficos).map(grafico => grafico.id);

    return Promise.all(
        graficosIds.map(id => 
            new Promise(resolve => {
                const canvas = document.getElementById(id);
                if (canvas) {
                    resolve(canvas.toDataURL());
                } else {
                    resolve(null);
                }
            })
        )
    );
}

function exportarReporte() {
    const titulo = document.querySelector('h3').innerText;
    const descripcion = document.querySelector('p').innerText;
    const tablasDatos = convertirTablaAJSON('.table');
    const imagenesURLs = Array.from(document.querySelectorAll('img.incluye-en-reporte')).map(img => img.src);

    convertirGraficasAImagenesBase64().then(graficasBase64 => {
        const graficas = [...graficasBase64, ...imagenesURLs].filter(g => g !== null);

        const reporteData = {
            titulo,
            descripcion,
            tablas: tablasDatos,
            graficas: graficas,
            csrfmiddlewaretoken: obtenerCSRFToken()
        };

        enviarReporteAlServidor(reporteData);
    });
}
function enviarReporteAlServidor(data) {
    const datosConGraficas = {
        titulo: data.titulo,
        descripcion: data.descripcion,
        tablas: data.tablas,
        graficas: data.graficas,
        csrfmiddlewaretoken: data.csrfmiddlewaretoken
    };

    fetch('/report/exportar_reporte/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': obtenerCSRFToken()
        },
        body: JSON.stringify(datosConGraficas)
    })
    .then(response => {
        const contentType = response.headers.get("Content-Type");
        if (!response.ok) {
            if (contentType.includes("application/json")) {
                return response.json().then(errorData => {
                    console.error('Detalle del error:', errorData);
                    throw new Error(`La solicitud falló: ${response.statusText} - Código de estado: ${response.status}`);
                });
            } else {
                return response.text().then(errorText => {
                    console.error('Respuesta de error no JSON:', errorText);
                    throw new Error(`La solicitud falló y no se devolvió JSON: ${response.statusText} - Código de estado: ${response.status}`);
                });
            }
        }
        return response.blob();
    })
    .then(data => {
        if (data instanceof Blob) {
            const url = window.URL.createObjectURL(data);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'reporte.docx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        }
    })
    .catch(error => {
        console.error('Error al exportar el reporte:', error);
        alert('Ocurrió un error al exportar el reporte. Verifica la consola para más detalles.');
    });
}