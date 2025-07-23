# Video Person Analysis & Retrieval Pipeline

Este proyecto permite detectar, trackear y analizar personas en videos, generar galerías de rostros, indexar embeddings faciales, realizar búsquedas de personas por imagen y obtener descripciones automáticas usando LLMs (Large Language Models). Todo el flujo está orientado a la extracción y análisis multimodal de personas en secuencias de video.

---

## **Flujo General del Proyecto**

1. **Detección y Tracking de Personas:**  
   - Se detectan personas frame a frame usando YOLOv8
   - Se trackea a cada persona con un identificador único (track_id) con SORT.

2. **Recorte y Gestión de Rostros:**  
   - De cada persona detectada, se recorta el rostro usando un detector facial (ej: RetinaFace).
   - Los recortes de rostro se guardan **con margen (padding)** para asegurar buena calidad en el embedding.

3. **Gestión de Galerías y Mejores Rostros:**  
   - Para cada track_id se guardan las N mejores imágenes de rostro y cuerpo.
   - Se genera una galería visual (8x8) de las mejores caras por video.

4. **Indexado y Embedding Facial:**  
   - Los mejores recortes de rostro son procesados con un modelo de embedding
   - Los embeddings se indexan usando FAISS para búsquedas rápidas.

5. **Búsqueda por Imagen (Query by Example):**  
   - Dada una imagen de rostro de consulta, se busca la persona más similar en el video.
   - Se retorna el track_id, los crops y los datos relevantes de la persona encontrada.

6. **Análisis y Resumen por LLM (LangChain/OpenAI):**  
   - Se usa un modelo LLM para generar descripciones automáticas, análisis o sumarización sobre la persona detectada, usando toda la información del pipeline.

---

## Cómo Ejecutar el Pipeline (`main.py`)

### Estructura Requerida de Directorios

Asegúrate de seguir esta estructura dentro del directorio raíz del proyecto:

```
data/
├── raw_videos/
│   └── tu_video.mp4
├── json/
├── crops/
├── faces/
├── bodies/
├── top_faces/
├── top_bodies/
└── search/
    └── tu_video/
        └── queries/
            └── rostro_consulta.jpg
```
Puede usar el siguiente comando, cambiando "tu_video" por el nombre del video a analizar
```
mkdir -p data/raw_videos \
         data/json \
         data/crops \
         data/top_faces \
         data/top_bodies \
         data/search/tu_video/queries
```
- **raw\_videos:** Coloca aquí los videos `.mp4` que deseas analizar.
- **search/{nombre\_video}/queries:** Coloca aquí las imágenes de rostro que deseas buscar dentro del video procesado. (Al procesar el video se guarda una carpeta con todas las caras en faces, para poder testear la busqueda mas facil, se pueden copiar imagenes de aqui a la carpeta de queries, ya que el embedding se realiza sobre la carpeta top_faces)

### 🖥️ Ejecución del Script

Ejecuta el script principal (`main.py`) desde la terminal con las siguientes opciones:

```bash
python3 main.py --video tu_video.mp4 --task [detect | postprocess]
```

#### Detalle de las Tareas (`--task`):

- `detect`: Detecta y trackea personas, guarda resultados en `data/json/persons_{nombre_video}.json`.

```bash
python3 main.py --video tu_video.mp4 --task detect
```

- `postprocess`: Crea embeddings, indexa rostros y realiza búsqueda por imagen de consulta. Los resultados visuales quedan guardados en `data/search/{nombre_video}/results/`.

```bash
python3 main.py --video tu_video.mp4 --task postprocess
```

### Dónde se guardan los Resultados

- **Detecciones y Tracking:** `data/json/persons_{nombre_video}.json`
- **Rostros y cuerpos:**
  - Rostros: `data/top_faces/{nombre_video}/`
  - Cuerpos: `data/top_bodies/{nombre_video}/`
- **Resultados búsqueda por imagen:**

```
data/search/{nombre_video}/results/
└── rostro_consulta/
    ├── matched_face.jpg
    ├── matched_body.jpg
    ├── trajectory.png
    ├── velocity.png
    ├── final_visualization.png
    ├── track_id.txt
    └── summary.json
```

---

📌 Asegúrate de instalar previamente todas las dependencias del proyecto indicadas en `requirements.txt`.

