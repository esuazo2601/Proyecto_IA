# Proyecto realizado por:

* Linna Cao 
* Patricio Covarrubias 
* Esteban Suazo
# Requisitos para ejecutar:
* Python versión 3.12
```
pip install gymnasium[toy-text] matplotlib numpy pygame
```

## ¿En qúe consiste el proyecto?
* Consistirá en el uso de varios algoritmos de búsqueda y de aprendizaje por refuerzo, 
implementados en el contexto de encontrar un camino hasta llegar a la casilla de destino
intentando evitar por el camino los obstáculos presentes (en este caso los pozos de hielo). De esta forma se pretende medir el desmepeño
de estos algoritmos implementados para evaluar su eficiencia.

* Preliminarmente se tiene pensado implementar y comparar los algoritmos:

Búsqueda no informada: DFS.
Búsqueda informada: A*. 
Aprendizaje por refuerzo: Q-Learning.

## Caracterización del problema a resolver
### Descripción del ambiente:

* Frozen Lake: Se trata de un ambiente observable, determinístico, secuencial, estático, discreto y de agente singular.
* Taxi: Ambiente observable, no determinista, secuencial, discreto, de agente singular y estático

### Representación del estado del juego

* En este caso la representación del ambiente del juego se trata de una matriz que muestra la configuración del mapa de forma que 
el caracter 'S' representa el inicio, la 'F' representa espacio caminable , la 'H' los agujeros de hielo y 'G' el destino. Ejemplo:

```
    "4x4":[
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ]

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]
```

Además se cuenta con una variable Score que cambia de 0 a 1 al llegar a la casilla final.

* En el caso del ambiente de Taxi, se representa de la siguiente manera:
```
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
```
Además se cuenta con una variable reward, que retorna -1 al moverse, +20 por dejar un pasajero y -10 por ejecutar las acciones de recojer o dejar pasajero en un lugar no permitido.


### Acciones que puede tomar el agente en Frozen Lake

* El espacio de acciones del agente se representa en un array de numeros enteros desde el 0 al 3, es decir en cada paso tomará la decisión
de una de estas 4 acciones (es discreto). Estas representan lo siguiente:

```
0 : Moverse a la izquierda, 1 : Moverse abajo, 2 : Moverse a la derecha, 3 : Moverse arriba
```


### Acciones que puede tomar el agente en Taxi

* El espacio de acciones del agente se representa en un array de numeros enteros desde el 0 al 5, es decir en cada paso tomará la decisión.

```
0 : Moverse abajo, 1 : Moverse arriba, 2 : Moverse a la derecha, 3 : Moverse izquierda, 4: Recojer pasajero, 5: Dejar pasajero
