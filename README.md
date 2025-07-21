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
   - Los mejores recortes de rostro son procesados con un modelo de embedding (ej: ArcFace, InsightFace).
   - Los embeddings se indexan usando FAISS para búsquedas rápidas.

5. **Búsqueda por Imagen (Query by Example):**  
   - Dada una imagen de rostro de consulta, se busca la persona más similar en el video.
   - Se retorna el track_id, los crops y los datos relevantes de la persona encontrada.

6. **Análisis y Resumen por LLM (LangChain/OpenAI):**  
   - Se usa un modelo LLM para generar descripciones automáticas, análisis o sumarización sobre la persona detectada, usando toda la información del pipeline.

---

