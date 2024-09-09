

from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from pycallgraph2 import Config
from pycallgraph2 import GlobbingFilter

import numpy as np
import torch
from einops import reduce

# Configuration de pycallgraph2 pour ignorer les appels à einops
config = Config()
config.trace_filter = GlobbingFilter(exclude=[
    'einops.*',  # Ignorer tout ce qui vient du module einops
    'torch.*'
])

graphviz = GraphvizOutput(output_file='trace.png')

# Fonction utilisant einops.reduce (sera ignorée par pycallgraph2)
def einops_operation():
    # Créer une matrice avec numpy (convertir en float pour éviter l'erreur)
    tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # Réduction en utilisant einops.reduce (moyenne sur la première dimension)
    #result = reduce(tensor, 'b c -> c', 'mean')
    result = tensor.mean(dim=0)  # Calculer la moyenne sur la dimension 0
    return result

# Code principal
def main():
    print("Début du traçage")
    
    # Appel de la fonction einops
    result = einops_operation()
    
    print("Résultat de la réduction avec einops :", result)
    print("Fin du traçage")

# Démarrage du traçage avec pycallgraph2
with PyCallGraph(output=graphviz, config=config):
    main()

# À la fin de l'exécution, une image trace.png sera générée sans les appels à einops
