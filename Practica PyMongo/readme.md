-Conexión a la base de datos: Nos conectamos, mediante pymongo, a la base de datos y a la colección que creamos para los ejercicios resueltos.

-Visualización sencilla de los datos: Generamos gráficos con matplotlib que muestren la distribución de valores de "borough" y "cuisine". Los posibles valores de cada variable y su número de registros asociados deben generarse mediante una consulta en pymongo.

-Inserción de datos: Insertamos todos los nuevos restaurantes almacenados en "new_restaurants.json".

-Añadir nuevos campos a los documentos: Creamos un valor raíz en cada documento con el nombre "avg_score", que contiene el valor medio de "puntuación" de ese restaurante. Si no se han encontrado valores para realizar este cálculo, ponemos en su lugar un valor Nulo. Se puede hacer de dos manera el cálculo, o bien con una query de Mongo o bien haciendo cada cálculo con Python en insertando los resultados.
