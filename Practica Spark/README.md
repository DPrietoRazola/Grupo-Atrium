Completa las celdas del notebook:
https://drive.google.com/file/d/1-boyqRGdLmqDKW_jK9sYFz0p39Wom7MI/view?usp=sharing

Los datos a utilizar son:
https://raw.githubusercontent.com/pratikbarjatya/spark-walmart-data-analysis-exercise/master/walmart_stock.csv

 
NOTA: En el caso de que no queráis instalar Spark, podéis usar Spark en Google Colab poniendo lo siguiente en la primera celda:
!pip install pyspark
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
