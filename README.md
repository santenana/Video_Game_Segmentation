## Genshin Segmentation 
En este Mini Laboratorio se hace uso de imágenes tomadas de un gameplay propio de un Video Juego para detectar a diferentes personajes y entornos como ejercicio académico y ver la eficiencia y consideraciones a la hora de hacer una segmentación de imágenes y videos.


## Modelo.py 
Para el modelo se usa Detectron, Roboflow y CUDA usando redes pre-entrenadas para la segmentacion de objetos, y el uso de los formatos de segmentacion y Data-Augmentation Roboflow para entrenar los modelos.

## Segmentation.ipynb 

En este notebook se muestra el paso a paso de como se entrena el modelo y se muestran los resultados de este, mostrando como el desempeño del modelo a la hora de hacer la segmentación

##  Consideraciones

Dentro del modelo se observa que para algunos NPC, como objetos del entorno el modelo no es capaz de hacer una buena segmentación, esto como tal se debe a la falta de datos, si bien como ejercicio académico se usaron una cantidad pequeña de datos, cerca de 20 imágenes para una mejora considerable en la segmentación se deben tomar mas muestras del personaje a segmentar, de los elementos de los entornos, de NPC, de enemigos de la flora y fauna del juego entre otros varios elementos de interés.

