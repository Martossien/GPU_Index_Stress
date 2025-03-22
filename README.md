 GPU_Index_Stress (GIS)

**GPU_Index_Stress (GIS)** est un outil tout-en-un pour résoudre les problèmes d'indexation GPU, effectuer des stress tests et benchmarker vos GPU NVIDIA avec PyTorch.

## Problématique

Dans les environnements multi-GPU, trois systèmes d'indexation coexistent et ne correspondent pas toujours :
- Index CUDA/NVML
- Index nvidia-smi
- Index PyTorch

Cette disparité cause de nombreux problèmes lorsqu'on tente d'utiliser un GPU spécifique. GIS résout ce problème en créant une table de correspondance fiable entre les différents systèmes d'indexation.

## Fonctionnalités

- **🔄 Mapping précis des index** : Détecte et harmonise les correspondances entre CUDA, nvidia-smi et PyTorch
- **🔥 Stress tests configurables** : Teste la stabilité et les performances de vos GPU
- **📊 Benchmarks intégrés** : Évalue les performances réelles (GFLOPS et bande passante mémoire)
- **🌡️ Surveillance des températures** : Surveille les températures pendant les tests
- **💾 Mise en cache des correspondances** : Évite des détections répétées chronophages

## Installation

### Prérequis

- Python 3.6+
- CUDA Toolkit
- Carte(s) graphique(s) NVIDIA

### Installation des dépendances

```bash
pip install -r requirements.txt

Utilisation rapide
Afficher la correspondance des index

python gis.py --list

Lancer un stress test sur tous les GPU ( attention consommation électrique )

python gis.py

Lancer un stress test sur un GPU spécifique

# Par index CUDA
python gis.py --cuda 2

# Par index PyTorch (recommandé)
python gis.py --pytorch 3

# Par index nvidia-smi
python gis.py --nvidia-smi 1

# Par identifiant PCI
python gis.py --pci 01:00.0

Options supplémentaires

# Utiliser FP16 pour le calcul (plus rapide sur GPU récents)
python gis.py --fp16

# Spécifier le niveau d'utilisation
python gis.py --util 80

# Forcer une nouvelle détection des index
python gis.py --force-detection

# Désactiver l'utilisation du cache
python gis.py --no-cache

Comment ça marche

    Détection des correspondances : L'outil active chaque GPU via PyTorch et observe l'activité pour identifier les correspondances
    Stress test : Utilise des multiplications matricielles intensives via PyTorch pour charger les GPU
    Benchmarks : Mesure les performances en GFLOPS et la bande passante mémoire

Contributions

Les contributions sont les bienvenues !

Licence

Ce projet est sous licence MIT

