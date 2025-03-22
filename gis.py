import torch
import threading
import time
import signal
import sys
import os
import json
import argparse
from datetime import datetime
import subprocess
import re
import numpy as np
from collections import defaultdict

# Vérifier si pynvml est disponible
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("AVERTISSEMENT: pynvml n'est pas installé. Installation recommandée pour la surveillance des températures:")
    print("pip install nvidia-ml-py")

class GPUStressTest:
    def __init__(self, tensor_size=10000, use_fp16=False, utilization=95, monitoring_interval=1, 
                 detection_duration=4, use_cached_mapping=True, force_detection=False, mapping_file=None):
        self.tensor_size = tensor_size
        self.use_fp16 = use_fp16
        self.utilization = min(max(10, utilization), 100) / 100.0
        self.monitoring_interval = monitoring_interval
        self.detection_duration = detection_duration  # Durée du test pour la détection d'activité
        self.use_cached_mapping = use_cached_mapping
        self.force_detection = force_detection
        self.mapping_file = mapping_file or "gpu_mapping_cache.json"
        
        self.running = True
        self.threads = []
        self.monitoring_thread = None
        self.start_time = None
        self.log_file = f"gpu_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.temp_log_file = f"gpu_temps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.max_temps = {}  # Pour stocker les températures maximales
        
        # S'assurer que NVML est disponible et initialisé
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_working = True
            except Exception as e:
                print(f"ERREUR d'initialisation NVML: {e}")
                self.nvml_working = False
        else:
            self.nvml_working = False
        
        # Dictionnaires pour stocker les informations GPU
        self.nvidia_smi_order = []  # Ordre des GPU dans nvidia-smi/nvitop
        self.cuda_info = {}         # Infos GPU par ID CUDA (indices)
        self.pci_to_cuda = {}       # Mapping PCI ID -> CUDA ID
        self.cuda_to_pci = {}       # Mapping CUDA ID -> PCI ID
        self.nvitop_to_cuda = {}    # Mapping indice nvitop -> CUDA ID
        
        # Mapping entre les indices PyTorch et les indices CUDA/NVML
        self.pytorch_to_cuda = {}   # Mapping indice PyTorch -> CUDA ID
        self.cuda_to_pytorch = {}   # Mapping indice CUDA -> PyTorch ID
        
        # Nouvelle structure pour stocker les résultats de benchmark
        self.benchmark_results = {}
        
        # Configuration de l'interruption
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Initialiser NVML et détecter les GPU
        self.initialize_and_detect_gpus()
            
        # Préparation du fichier CSV pour les températures
        with open(self.temp_log_file, "w") as f:
            header = "Timestamp"
            for i in range(len(self.cuda_info)):
                header += f",GPU{i}_Temp,GPU{i}_Mem_Temp,GPU{i}_Util,GPU{i}_Mem_Util"
            f.write(header + "\n")
    
    def initialize_and_detect_gpus(self):
        """Initialiser NVML et détecter tous les GPU avec leurs informations"""
        # Détection des GPUs via CUDA
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            print("Aucun GPU compatible CUDA n'a été détecté!")
            sys.exit(1)
                
        # Récupérer l'ordre des GPU dans nvidia-smi (comme nvitop)
        self.get_nvidia_smi_order()
        
        # Récupérer les informations pour chaque GPU CUDA en utilisant NVML directement
        print(f"Détection de {self.num_gpus} GPU(s) CUDA:")
        
        # Récupérer les informations de base pour chaque GPU
        for cuda_idx in range(self.num_gpus):
            # Utiliser NVML pour obtenir le nom correct et le PCI ID
            if self.nvml_working:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)
                    
                    # Récupération du nom
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode('utf-8')
                    
                    # Récupération du PCI ID
                    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                    pci_id = pci_info.busId
                    if isinstance(pci_id, bytes):
                        pci_id = pci_id.decode('utf-8')
                        
                    # Récupération de la mémoire
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_total = mem_info.total / (1024 ** 3)  # Convert to GB
                    
                    # Température actuelle
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temp_info = f", {temp}°C"
                except Exception as e:
                    print(f"Erreur NVML pour GPU {cuda_idx}: {e}")
                    # Fallback à PyTorch
                    gpu_name = torch.cuda.get_device_name(cuda_idx)
                    pci_id = self.get_gpu_pci_id_via_smi(cuda_idx)
                    vram_total = torch.cuda.get_device_properties(cuda_idx).total_memory / (1024 ** 3)
                    temp_info = ""
            else:
                # Fallback à PyTorch
                gpu_name = torch.cuda.get_device_name(cuda_idx)
                pci_id = self.get_gpu_pci_id_via_smi(cuda_idx)
                vram_total = torch.cuda.get_device_properties(cuda_idx).total_memory / (1024 ** 3)
                temp_info = ""
            
            # Trouver l'indice nvidia-smi correspondant
            nvitop_idx = self.find_nvitop_index(pci_id)
            
            # Stocker les infos du GPU
            self.cuda_info[cuda_idx] = {
                'name': gpu_name,
                'pci_id': pci_id,
                'nvitop_idx': nvitop_idx if nvitop_idx is not None else None,
                'pytorch_idx': None,  # Sera rempli plus tard
                'vram_gb': round(vram_total, 1),
                'matrix_perf': None,  # Pour le benchmark de multiplication de matrices
                'memory_bw': None,    # Pour le benchmark de bande passante mémoire
                'perf_index': None    # Indice de performance composite
            }
            
            # Construire les mappings
            self.cuda_to_pci[cuda_idx] = pci_id
            self.pci_to_cuda[pci_id] = cuda_idx
            
            if nvitop_idx is not None:
                self.nvitop_to_cuda[nvitop_idx] = cuda_idx
            
            print(f"  GPU {cuda_idx}: {gpu_name}")
            print(f"     - PCI ID: {pci_id}")
            print(f"     - VRAM: {round(vram_total, 1)} GB")
            print(f"     - Indice nvidia-smi: {nvitop_idx if nvitop_idx is not None else 'inconnu'}{temp_info}")
        
        # Essayer de charger les correspondances depuis un fichier cache, sinon les détecter
        if not self.load_pytorch_mapping_from_cache() or self.force_detection:
            # NOUVELLE MÉTHODE AMÉLIORÉE: Détecter les index PyTorch par activité directe avec NVML
            if self.nvml_working:
                self.detect_pytorch_indices_via_activity_nvml()
            else:
                self.log("AVERTISSEMENT: NVML n'est pas disponible, utilisation de la méthode alternative pour détecter les index PyTorch")
                self.detect_pytorch_indices_via_activity_smi()
            
            # Sauvegarder les correspondances dans le fichier cache
            self.save_pytorch_mapping_to_cache()
        
        # Exécuter les benchmarks pour évaluer les performances
        self.run_benchmarks()
        
        # Afficher un tableau récapitulatif
        self.print_gpu_mapping()

    def load_pytorch_mapping_from_cache(self):
        """
        Charge les correspondances entre indices PyTorch et CUDA depuis un fichier cache.
        Retourne True si le chargement a réussi et les correspondances sont cohérentes.
        """
        if not self.use_cached_mapping:
            return False
            
        try:
            if not os.path.exists(self.mapping_file):
                return False
                
            with open(self.mapping_file, 'r') as f:
                cache_data = json.load(f)
            
            # Vérifier la cohérence du cache
            if cache_data.get('num_gpus') != self.num_gpus:
                self.log(f"Le nombre de GPU a changé ({cache_data.get('num_gpus')} -> {self.num_gpus}), nouveau mapping requis")
                return False
            
            # Vérifier que les PCI IDs sont identiques
            pci_match = True
            for cuda_idx, pci_id in cache_data.get('cuda_to_pci', {}).items():
                cuda_idx = int(cuda_idx)  # JSON stocke les clés comme des strings
                if cuda_idx not in self.cuda_to_pci or self.cuda_to_pci[cuda_idx] != pci_id:
                    pci_match = False
                    break
            
            if not pci_match:
                self.log("La configuration GPU a changé, nouveau mapping requis")
                return False
            
            # Charger les correspondances
            self.pytorch_to_cuda = {int(k): int(v) for k, v in cache_data.get('pytorch_to_cuda', {}).items()}
            self.cuda_to_pytorch = {int(k): int(v) for k, v in cache_data.get('cuda_to_pytorch', {}).items()}
            
            # Mettre à jour les informations dans cuda_info
            for cuda_idx in range(self.num_gpus):
                self.cuda_info[cuda_idx]['pytorch_idx'] = self.cuda_to_pytorch.get(cuda_idx, None)
            
            self.log(f"Correspondances PyTorch chargées depuis le cache: {len(self.pytorch_to_cuda)}/{self.num_gpus} GPU mappés")
            return True
            
        except Exception as e:
            self.log(f"Erreur lors du chargement du cache de mapping: {e}")
            return False
    
    def save_pytorch_mapping_to_cache(self):
        """Sauvegarde les correspondances entre indices PyTorch et CUDA dans un fichier cache."""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'num_gpus': self.num_gpus,
                'pytorch_to_cuda': {str(k): str(v) for k, v in self.pytorch_to_cuda.items()},
                'cuda_to_pytorch': {str(k): str(v) for k, v in self.cuda_to_pytorch.items()},
                'cuda_to_pci': {str(k): v for k, v in self.cuda_to_pci.items()},
                'pci_to_cuda': {k: str(v) for k, v in self.pci_to_cuda.items()}
            }
            
            with open(self.mapping_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            self.log(f"Correspondances PyTorch sauvegardées dans {self.mapping_file}")
            
        except Exception as e:
            self.log(f"Erreur lors de la sauvegarde du cache de mapping: {e}")
    
    def detect_pytorch_indices_via_activity_nvml(self):
        """
        Détecte les correspondances entre indices PyTorch et CUDA en observant 
        l'activité des GPU via NVML (plus précis et plus rapide que nvidia-smi).
        """
        self.log("Détection précise des index PyTorch par test d'activité via NVML...")
        
        # Pour stocker les correspondances détectées
        pytorch_to_cuda_temp = {}
        
        # Obtenir les handles NVML pour tous les GPU
        nvml_handles = {}
        for cuda_idx in range(self.num_gpus):
            try:
                pci_id = self.cuda_info[cuda_idx]['pci_id']
                handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_id.encode('utf-8'))
                nvml_handles[cuda_idx] = handle
            except Exception as e:
                self.log(f"  Erreur lors de l'obtention du handle NVML pour GPU {cuda_idx}: {e}")
        
        # Tester chaque index PyTorch possible
        for pytorch_idx in range(self.num_gpus):
            self.log(f"  Test de l'index PyTorch {pytorch_idx}...")
            
            try:
                # Récupérer les mesures initiales pour tous les GPU
                initial_utils = {}
                initial_mems = {}
                for cuda_idx, handle in nvml_handles.items():
                    try:
                        # Utilisation GPU
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        # Mémoire utilisée
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used = mem_info.used / (1024 * 1024)  # en MB
                        
                        initial_utils[cuda_idx] = util
                        initial_mems[cuda_idx] = mem_used
                    except Exception as e:
                        self.log(f"    Erreur mesure initiale GPU {cuda_idx}: {e}")
                        initial_utils[cuda_idx] = 0
                        initial_mems[cuda_idx] = 0
                
                # Nettoyer la mémoire
                torch.cuda.empty_cache()
                # Pause pour stabiliser
                time.sleep(0.5)
                
                # Mesures pendant le test
                utils_during_test = defaultdict(list)
                mems_during_test = defaultdict(list)
                
                # Lancer une opération intensive sur le GPU avec index PyTorch spécifique
                self.log(f"    Lancement des opérations sur PyTorch {pytorch_idx}...")
                
                # Démarrer un thread pour surveiller l'activité pendant le test
                stop_monitoring = False
                
                def monitor_activity():
                    start_time = time.time()
                    sample_count = 0
                    
                    while not stop_monitoring:
                        for cuda_idx, handle in nvml_handles.items():
                            try:
                                # Utilisation GPU
                                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                                # Mémoire utilisée
                                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                mem_used = mem_info.used / (1024 * 1024)  # en MB
                                
                                utils_during_test[cuda_idx].append(util)
                                mems_during_test[cuda_idx].append(mem_used)
                            except Exception:
                                pass
                        
                        sample_count += 1
                        # Ajuster l'intervalle d'échantillonnage (4-5 échantillons par seconde)
                        time.sleep(0.2)
                
                # Démarrer le thread de surveillance
                monitor_thread = threading.Thread(target=monitor_activity)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # Exécuter les opérations sur le GPU
                try:
                    with torch.cuda.device(pytorch_idx):
                        # Allouer un grand tenseur pour consommer de la mémoire
                        memory_tensor = torch.zeros((2000, 2000), device=f"cuda:{pytorch_idx}")
                        
                        # Opérations intensives pour augmenter l'utilisation
                        start_time = time.time()
                        while time.time() - start_time < self.detection_duration:
                            a = torch.rand((3000, 3000), device=f"cuda:{pytorch_idx}")
                            b = torch.rand((3000, 3000), device=f"cuda:{pytorch_idx}")
                            c = torch.matmul(a, b)
                            d = torch.sigmoid(c)
                            result = d.sum().item()  # Forcer le calcul
                except Exception as e:
                    self.log(f"    Erreur lors des opérations sur PyTorch {pytorch_idx}: {e}")
                
                # Arrêter la surveillance
                stop_monitoring = True
                monitor_thread.join(timeout=1.0)
                
                # Attendre que l'activité se stabilise
                time.sleep(0.5)
                
                # Récupérer les mesures finales
                final_utils = {}
                final_mems = {}
                for cuda_idx, handle in nvml_handles.items():
                    try:
                        # Utilisation GPU
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        # Mémoire utilisée
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used = mem_info.used / (1024 * 1024)  # en MB
                        
                        final_utils[cuda_idx] = util
                        final_mems[cuda_idx] = mem_used
                    except Exception:
                        final_utils[cuda_idx] = 0
                        final_mems[cuda_idx] = 0
                
                # Déterminer quel GPU a montré la plus forte activité
                max_util_increase = -1
                max_mem_increase = -1
                detected_cuda_idx = None
                detection_confidence = 0
                
                # Analyser les mesures pour chaque GPU
                for cuda_idx in nvml_handles.keys():
                    # Calcul des maximum observés pendant le test
                    max_util = max(utils_during_test[cuda_idx]) if utils_during_test[cuda_idx] else 0
                    max_mem = max(mems_during_test[cuda_idx]) if mems_during_test[cuda_idx] else 0
                    
                    # Calcul des augmentations
                    util_increase = max_util - initial_utils[cuda_idx]
                    mem_increase = max_mem - initial_mems[cuda_idx]
                    
                    self.log(f"    GPU CUDA {cuda_idx}: Util max {max_util}% (+{util_increase}%), Mém max {max_mem:.0f}MB (+{mem_increase:.0f}MB)")
                    
                    # Détection basée sur l'utilisation (seuil de 45%)
                    if util_increase >= 45 and util_increase > max_util_increase:
                        max_util_increase = util_increase
                        detected_cuda_idx = cuda_idx
                        detection_confidence = 1.0
                    
                    # Si aucune détection forte par utilisation, essayer la mémoire (seuil de 300MB)
                    elif detected_cuda_idx is None and mem_increase > 300 and mem_increase > max_mem_increase:
                        max_mem_increase = mem_increase
                        detected_cuda_idx = cuda_idx
                        detection_confidence = 0.7  # Confiance plus faible
                
                if detected_cuda_idx is not None:
                    confidence_str = "élevée" if detection_confidence > 0.8 else "modérée"
                    self.log(f"  ✓ PyTorch {pytorch_idx} correspond à CUDA {detected_cuda_idx} (confiance {confidence_str})")
                    pytorch_to_cuda_temp[pytorch_idx] = detected_cuda_idx
                else:
                    self.log(f"  ✗ Aucune correspondance claire trouvée pour PyTorch {pytorch_idx}")
                
                # Nettoyer la mémoire
                torch.cuda.empty_cache()
                # Pause pour que l'activité redescende avant le prochain test
                time.sleep(2)
                
            except Exception as e:
                self.log(f"  ✗ Erreur lors du test de l'index PyTorch {pytorch_idx}: {e}")
        
        # Vérification de cohérence (un CUDA ne devrait pas être associé à plusieurs PyTorch)
        cuda_to_pytorch_count = {}
        for pt_idx, cuda_idx in pytorch_to_cuda_temp.items():
            cuda_to_pytorch_count[cuda_idx] = cuda_to_pytorch_count.get(cuda_idx, 0) + 1
        
        # Résoudre les conflits potentiels
        conflicts = [cuda_idx for cuda_idx, count in cuda_to_pytorch_count.items() if count > 1]
        if conflicts:
            self.log(f"Attention: {len(conflicts)} GPU CUDA ont des correspondances multiples. Résolution...")
            
            # Pour chaque conflit, conserver la correspondance avec l'index PyTorch le plus bas
            for cuda_idx in conflicts:
                conflicting_pt_indices = [pt_idx for pt_idx, c_idx in pytorch_to_cuda_temp.items() if c_idx == cuda_idx]
                best_pt_idx = min(conflicting_pt_indices)
                
                for pt_idx in conflicting_pt_indices:
                    if pt_idx != best_pt_idx:
                        self.log(f"  Conflit résolu: PyTorch {pt_idx} -> CUDA {cuda_idx} supprimé (gardé PyTorch {best_pt_idx})")
                        pytorch_to_cuda_temp.pop(pt_idx)
        
        # Mettre à jour les mappings définitifs
        self.pytorch_to_cuda = pytorch_to_cuda_temp
        self.cuda_to_pytorch = {cuda_idx: pt_idx for pt_idx, cuda_idx in pytorch_to_cuda_temp.items()}
        
        # Mettre à jour les informations dans cuda_info
        for cuda_idx in range(self.num_gpus):
            self.cuda_info[cuda_idx]['pytorch_idx'] = self.cuda_to_pytorch.get(cuda_idx, None)
        
        self.log(f"Détection des index PyTorch terminée. {len(self.pytorch_to_cuda)}/{self.num_gpus} GPU correctement identifiés.")

    def detect_pytorch_indices_via_activity_smi(self):
        """
        Méthode alternative utilisant nvidia-smi quand NVML n'est pas disponible.
        Moins précise mais fonctionne sans pynvml.
        """
        self.log("Détection des index PyTorch par test d'activité via nvidia-smi (méthode alternative)...")
        
        # Pour stocker les correspondances détectées
        pytorch_to_cuda_temp = {}
        
        # Tester chaque index PyTorch possible
        for pytorch_idx in range(self.num_gpus):
            self.log(f"  Test de l'index PyTorch {pytorch_idx}...")
            
            try:
                # Mesures initiales via nvidia-smi
                initial_utils = self.get_all_gpu_utilizations_smi()
                initial_mems = self.get_all_gpu_memory_smi()
                
                # Nettoyer la mémoire
                torch.cuda.empty_cache()
                time.sleep(0.5)
                
                # Exécuter des opérations intensives sur le GPU
                with torch.cuda.device(pytorch_idx):
                    # Allouer un grand tenseur pour consommer de la mémoire
                    memory_tensor = torch.zeros((2000, 2000), device=f"cuda:{pytorch_idx}")
                    
                    # Opérations intensives
                    start_time = time.time()
                    while time.time() - start_time < self.detection_duration:
                        a = torch.rand((3000, 3000), device=f"cuda:{pytorch_idx}")
                        b = torch.rand((3000, 3000), device=f"cuda:{pytorch_idx}")
                        c = torch.matmul(a, b)
                        d = torch.sigmoid(c)
                        result = d.sum().item()
                
                # Plusieurs mesures pendant l'activité
                utils_during = []
                mems_during = []
                
                # Prendre plusieurs mesures
                for _ in range(3):
                    utils_during.append(self.get_all_gpu_utilizations_smi())
                    mems_during.append(self.get_all_gpu_memory_smi())
                    time.sleep(0.3)
                
                # Calculer les valeurs maximales observées
                max_utils = []
                max_mems = []
                
                for cuda_idx in range(self.num_gpus):
                    max_util = max([utils[cuda_idx] for utils in utils_during]) if utils_during else 0
                    max_mem = max([mems[cuda_idx] for mems in mems_during]) if mems_during else 0
                    max_utils.append(max_util)
                    max_mems.append(max_mem)
                
                # Mesures finales
                final_utils = self.get_all_gpu_utilizations_smi()
                final_mems = self.get_all_gpu_memory_smi()
                
                # Déterminer quel GPU a montré la plus forte activité
                max_util_increase = -1
                max_mem_increase = -1
                detected_cuda_idx = None
                
                for cuda_idx in range(self.num_gpus):
                    util_increase = max_utils[cuda_idx] - initial_utils[cuda_idx]
                    mem_increase = max_mems[cuda_idx] - initial_mems[cuda_idx]
                    
                    self.log(f"    GPU CUDA {cuda_idx}: Util max {max_utils[cuda_idx]}% (+{util_increase}%), Mém max {max_mems[cuda_idx]}MB (+{mem_increase}MB)")
                    
                    # Détection par utilisation (seuil de 45%)
                    if util_increase >= 45 and util_increase > max_util_increase:
                        max_util_increase = util_increase
                        detected_cuda_idx = cuda_idx
                    
                    # Si pas de détection forte par utilisation, essayer la mémoire
                    elif detected_cuda_idx is None and mem_increase > 300 and mem_increase > max_mem_increase:
                        max_mem_increase = mem_increase
                        detected_cuda_idx = cuda_idx
                
                if detected_cuda_idx is not None:
                    self.log(f"  ✓ PyTorch {pytorch_idx} correspond à CUDA {detected_cuda_idx}")
                    pytorch_to_cuda_temp[pytorch_idx] = detected_cuda_idx
                else:
                    self.log(f"  ✗ Aucune correspondance claire trouvée pour PyTorch {pytorch_idx}")
                
                # Nettoyer la mémoire
                torch.cuda.empty_cache()
                # Pause avant le prochain test
                time.sleep(2)
                
            except Exception as e:
                self.log(f"  ✗ Erreur lors du test de l'index PyTorch {pytorch_idx}: {e}")
                import traceback
                self.log(traceback.format_exc())
        
        # Vérification de cohérence (un CUDA ne devrait pas être associé à plusieurs PyTorch)
        cuda_to_pytorch_count = {}
        for pt_idx, cuda_idx in pytorch_to_cuda_temp.items():
            cuda_to_pytorch_count[cuda_idx] = cuda_to_pytorch_count.get(cuda_idx, 0) + 1
        
        # Résoudre les conflits potentiels
        conflicts = [cuda_idx for cuda_idx, count in cuda_to_pytorch_count.items() if count > 1]
        if conflicts:
            self.log(f"Attention: {len(conflicts)} GPU CUDA ont des correspondances multiples. Résolution...")
            
            # Pour chaque conflit, conserver la correspondance avec l'index PyTorch le plus bas
            for cuda_idx in conflicts:
                conflicting_pt_indices = [pt_idx for pt_idx, c_idx in pytorch_to_cuda_temp.items() if c_idx == cuda_idx]
                best_pt_idx = min(conflicting_pt_indices)
                
                for pt_idx in conflicting_pt_indices:
                    if pt_idx != best_pt_idx:
                        self.log(f"  Conflit résolu: PyTorch {pt_idx} -> CUDA {cuda_idx} supprimé (gardé PyTorch {best_pt_idx})")
                        pytorch_to_cuda_temp.pop(pt_idx)
        
        # Mettre à jour les mappings définitifs
        self.pytorch_to_cuda = pytorch_to_cuda_temp
        self.cuda_to_pytorch = {cuda_idx: pt_idx for pt_idx, cuda_idx in pytorch_to_cuda_temp.items()}
        
        # Mettre à jour les informations dans cuda_info
        for cuda_idx in range(self.num_gpus):
            self.cuda_info[cuda_idx]['pytorch_idx'] = self.cuda_to_pytorch.get(cuda_idx, None)
        
        self.log(f"Détection des index PyTorch terminée. {len(self.pytorch_to_cuda)}/{self.num_gpus} GPU correctement identifiés.")
    
    def get_all_gpu_utilizations_smi(self):
        """
        Récupère l'utilisation actuelle de tous les GPU via nvidia-smi.
        Retourne une liste d'utilisation en pourcentage pour chaque GPU CUDA.
        """
        utilizations = [0] * self.num_gpus
        
        try:
            # Exécuter nvidia-smi pour obtenir les utilisations de GPU
            cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split()).decode('utf-8').strip()
            
            # Récupérer les valeurs d'utilisation
            utils = [int(line.strip()) for line in output.split('\n')]
            
            # Correspondance entre l'ordre nvidia-smi et les index CUDA
            for nvitop_idx, util in enumerate(utils):
                if nvitop_idx in self.nvitop_to_cuda:
                    cuda_idx = self.nvitop_to_cuda[nvitop_idx]
                    utilizations[cuda_idx] = util
        except Exception as e:
            self.log(f"Erreur lors de la récupération des utilisations GPU: {e}")
        
        return utilizations
    
    def get_all_gpu_memory_smi(self):
        """
        Récupère la mémoire actuellement utilisée de tous les GPU via nvidia-smi.
        Retourne une liste de mémoire utilisée en MB pour chaque GPU CUDA.
        """
        memory_used = [0] * self.num_gpus
        
        try:
            # Exécuter nvidia-smi pour obtenir la mémoire utilisée
            cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split()).decode('utf-8').strip()
            
            # Récupérer les valeurs de mémoire
            mems = [int(line.strip()) for line in output.split('\n')]
            
            # Correspondance entre l'ordre nvidia-smi et les index CUDA
            for nvitop_idx, mem in enumerate(mems):
                if nvitop_idx in self.nvitop_to_cuda:
                    cuda_idx = self.nvitop_to_cuda[nvitop_idx]
                    memory_used[cuda_idx] = mem
        except Exception as e:
            self.log(f"Erreur lors de la récupération de la mémoire GPU: {e}")
        
        return memory_used
    
    def run_benchmarks(self):
        """Exécute les benchmarks de performance sur tous les GPU détectés"""
        self.log("Exécution des benchmarks de performance...")
        
        # Référence pour normalisation (RTX 3090 = 100)
        reference_matrix_perf = 185.0  # GFLOPS estimés pour RTX 3090 sur le benchmark matrice
        reference_memory_bw = 760.0    # GB/s estimés pour RTX 3090 sur le benchmark mémoire
        
        # Valeurs par défaut pour les GPU qui ne peuvent pas être benchmarkés
        default_matrix_perf = 100.0
        default_memory_bw = 500.0
        default_perf_index = 100.0
        
        for cuda_idx, info in self.cuda_info.items():
            self.log(f"Benchmark du GPU {cuda_idx}: {info['name']}...")
            
            # Récupérer l'index PyTorch correspondant
            pytorch_idx = info['pytorch_idx']
            
            if pytorch_idx is not None:
                try:
                    # Benchmark de multiplication de matrices mixte
                    matrix_perf = self.benchmark_gpu_matrix_mult(pytorch_idx)
                    
                    # Benchmark de bande passante mémoire
                    memory_bw = self.benchmark_gpu_memory_bandwidth(pytorch_idx)
                    
                    # Calcul de l'indice de performance composite (70% matrices, 30% mémoire)
                    matrix_norm = (matrix_perf / reference_matrix_perf) * 100
                    memory_norm = (memory_bw / reference_memory_bw) * 100
                    perf_index = 0.7 * matrix_norm + 0.3 * memory_norm
                    
                    # Arrondir à une décimale
                    matrix_perf = round(matrix_perf, 1)
                    memory_bw = round(memory_bw, 1)
                    perf_index = round(perf_index, 1)
                    
                    # Stocker les résultats
                    self.cuda_info[cuda_idx]['matrix_perf'] = matrix_perf
                    self.cuda_info[cuda_idx]['memory_bw'] = memory_bw
                    self.cuda_info[cuda_idx]['perf_index'] = perf_index
                    
                    self.log(f"  ✓ Benchmark terminé: Perf Matrice={matrix_perf} GFLOPS, Bande Passante={memory_bw} GB/s, Index={perf_index}")
                    
                except Exception as e:
                    self.log(f"  ✗ Erreur lors du benchmark: {e}")
                    # Valeurs par défaut en cas d'erreur
                    self.cuda_info[cuda_idx]['matrix_perf'] = default_matrix_perf
                    self.cuda_info[cuda_idx]['memory_bw'] = default_memory_bw
                    self.cuda_info[cuda_idx]['perf_index'] = default_perf_index
            else:
                self.log(f"  ✗ Pas d'index PyTorch associé, benchmark impossible")
                # Valeurs par défaut si pas d'index PyTorch
                self.cuda_info[cuda_idx]['matrix_perf'] = default_matrix_perf
                self.cuda_info[cuda_idx]['memory_bw'] = default_memory_bw
                self.cuda_info[cuda_idx]['perf_index'] = default_perf_index
    
    def benchmark_gpu_matrix_mult(self, pytorch_idx):
        """
        Benchmark de multiplication de matrices mixte (FP32/FP16)
        Retourne les GFLOPS mesurés
        """
        # Nettoyer la mémoire avant le benchmark
        torch.cuda.empty_cache()
        
        # Paramètres du benchmark
        matrix_size = 4096
        warmup_iterations = 5
        measure_iterations = 10
        
        # Créer les tenseurs pour le benchmark
        with torch.cuda.device(pytorch_idx):
            # Tenseurs FP32
            tensor_a_fp32 = torch.rand(matrix_size, matrix_size, device=f"cuda:{pytorch_idx}")
            tensor_b_fp32 = torch.rand(matrix_size, matrix_size, device=f"cuda:{pytorch_idx}")
            
            # Tenseurs FP16
            tensor_a_fp16 = torch.rand(matrix_size, matrix_size, device=f"cuda:{pytorch_idx}", dtype=torch.float16)
            tensor_b_fp16 = torch.rand(matrix_size, matrix_size, device=f"cuda:{pytorch_idx}", dtype=torch.float16)
            
            # Warmup pour stabiliser les mesures
            for _ in range(warmup_iterations):
                _ = torch.matmul(tensor_a_fp32, tensor_b_fp32)
                _ = torch.matmul(tensor_a_fp16, tensor_b_fp16)
            
            # Synchroniser pour s'assurer que toutes les opérations précédentes sont terminées
            torch.cuda.synchronize()
            
            # Mesure des performances
            start_time = time.time()
            
            for _ in range(measure_iterations):
                # 50% FP32, 50% FP16
                _ = torch.matmul(tensor_a_fp32, tensor_b_fp32)
                _ = torch.matmul(tensor_a_fp16, tensor_b_fp16)
            
            # Synchroniser avant de mesurer le temps écoulé
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Calcul des GFLOPS
            elapsed_time = end_time - start_time
            # Formule: 2 * N^3 opérations pour une multiplication de matrices NxN
            # x2 pour les deux types de multiplication (FP32 + FP16)
            # x nombre d'itérations / temps en secondes / 10^9 (pour avoir des GFLOPS)
            total_ops = 2 * 2 * matrix_size**3 * measure_iterations
            gflops = total_ops / elapsed_time / 1e9
            
            return gflops
    
    def benchmark_gpu_memory_bandwidth(self, pytorch_idx):
        """
        Benchmark de bande passante mémoire GPU
        Retourne la bande passante mesurée en GB/s
        """
        # Nettoyer la mémoire avant le benchmark
        torch.cuda.empty_cache()
        
        # Paramètres du benchmark
        block_sizes = [16 * 1024**2, 64 * 1024**2, 256 * 1024**2]  # Différentes tailles de blocs (en éléments)
        warmup_iterations = 3
        measure_iterations = 10
        
        total_bandwidth = 0.0
        
        with torch.cuda.device(pytorch_idx):
            for block_size in block_sizes:
                # Créer les tenseurs source et destination
                src_tensor = torch.rand(block_size, device=f"cuda:{pytorch_idx}")
                dst_tensor = torch.zeros(block_size, device=f"cuda:{pytorch_idx}")
                
                # Warmup
                for _ in range(warmup_iterations):
                    dst_tensor.copy_(src_tensor)
                
                # Synchroniser
                torch.cuda.synchronize()
                
                # Mesure
                start_time = time.time()
                
                for _ in range(measure_iterations):
                    dst_tensor.copy_(src_tensor)
                
                # Synchroniser avant de mesurer le temps
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Calcul de la bande passante pour cette taille de bloc
                elapsed_time = end_time - start_time
                # bytes transférés = taille du bloc * sizeof(float32) * nombre d'itérations * 2 (lecture + écriture)
                bytes_transferred = block_size * 4 * measure_iterations * 2
                bandwidth_gb_s = bytes_transferred / elapsed_time / 1e9
                
                total_bandwidth += bandwidth_gb_s
            
            # Moyenne des bandes passantes pour différentes tailles de blocs
            avg_bandwidth = total_bandwidth / len(block_sizes)
            
            return avg_bandwidth
    
    def get_nvidia_smi_order(self):
        """Récupérer l'ordre des GPU dans nvidia-smi par leur ID PCI"""
        try:
            # Exécuter nvidia-smi pour obtenir les identifiants PCI dans l'ordre affiché
            cmd = "nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader"
            output = subprocess.check_output(cmd.split()).decode('utf-8').strip()
            
            # Stocker l'ordre des identifiants PCI
            self.nvidia_smi_order = [line.strip() for line in output.split('\n')]
            
            # Format complet pour chaque identifiant
            for i in range(len(self.nvidia_smi_order)):
                if not self.nvidia_smi_order[i].startswith("0000"):
                    self.nvidia_smi_order[i] = f"0000{self.nvidia_smi_order[i]}"
                    
        except Exception as e:
            print(f"AVERTISSEMENT: Impossible de déterminer l'ordre nvidia-smi: {e}")
            self.nvidia_smi_order = []
    
    def find_nvitop_index(self, pci_id):
        """Trouver l'indice dans nvitop/nvidia-smi correspondant à un identifiant PCI"""
        if not self.nvidia_smi_order:
            return None
        
        try:
            return self.nvidia_smi_order.index(pci_id)
        except ValueError:
            # Essayer avec le format court
            short_pci = pci_id.split(':')[-2] + ':' + pci_id.split(':')[-1]
            try:
                for i, smi_pci in enumerate(self.nvidia_smi_order):
                    if smi_pci.endswith(short_pci):
                        return i
            except:
                pass
        
        return None
        
    def get_gpu_pci_id_via_smi(self, cuda_idx):
        """Récupérer l'identifiant PCI via nvidia-smi pour un GPU donné"""
        try:
            cmd = f"nvidia-smi -i {cuda_idx} --query-gpu=pci.bus_id --format=csv,noheader"
            output = subprocess.check_output(cmd.split()).decode('utf-8').strip()
            pci_id = output.strip()
            
            # S'assurer que l'identifiant est au format complet
            if not pci_id.startswith("0000"):
                pci_id = f"0000{pci_id}"
                
            return pci_id
        except:
            pass
            
        # Si on ne peut pas obtenir le PCI ID
        return f"UNKNOWN_{cuda_idx}"
    
    def print_gpu_mapping(self):
        """Afficher un tableau récapitulatif des mappings GPU avec les indices de performance"""
        # Déterminer les largeurs des colonnes en fonction des valeurs maximales
        model_width = max(len(info['name']) for info in self.cuda_info.values()) + 2
        model_width = max(model_width, 25)  # Minimum pour "NVIDIA GeForce RTX 4060 Ti"
        
        # En-tête du tableau
        header_line = "=== TABLEAU DE CORRESPONDANCE DES GPU (VALIDÉ) ==="
        print("\n" + header_line)
        
        # Ligne de séparation avec longueur dynamique
        separator_line = "+" + "-" * 6 + "+" + "-" * 8 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * model_width + "+" + "-" * 10 + "+" + "-" * 15 + "+" + "-" * 15 + "+" + "-" * 13 + "+"
        print(separator_line)
        
        # En-têtes des colonnes
        print(f"| {'CUDA':<4} | {'PyTorh':<6} | {'nvidiasmi':<9} | {'PCI ID':<8} | {'Modèle':<{model_width-2}} | {'VRAM(GB)':<8} | {'Perf Matrice':<13} | {'Bande Passant':<13} | {'Perf Index':<11} |")
        print(f"| {'':<4} | {'':<6} | {'':<9} | {'':<8} | {'':<{model_width-2}} | {'':<8} | {'(GFLOPS)':<13} | {'(GB/s)':<13} | {'':<11} |")
        print(separator_line)
        
        # Lister tous les GPU CUDA dans l'ordre des indices nvidia-smi
        sorted_by_nvitop = sorted(
            [(cuda_idx, info) for cuda_idx, info in self.cuda_info.items() if info['nvitop_idx'] is not None],
            key=lambda x: x[1]['nvitop_idx']
        )
        
        # Ajouter les GPU sans indice nvidia-smi à la fin
        sorted_by_nvitop.extend(
            [(cuda_idx, info) for cuda_idx, info in self.cuda_info.items() if info['nvitop_idx'] is None]
        )
        
        # Afficher chaque GPU
        for cuda_idx, info in sorted_by_nvitop:
            nvitop_idx = info['nvitop_idx'] if info['nvitop_idx'] is not None else "?"
            pytorch_idx = info['pytorch_idx'] if info['pytorch_idx'] is not None else "?"
            pci_id = info['pci_id'].split(':')[-2] + ':' + info['pci_id'].split(':')[-1]  # Format court
            model = info['name']
            vram = info['vram_gb']
            matrix_perf = info['matrix_perf'] if info['matrix_perf'] is not None else "N/A"
            memory_bw = info['memory_bw'] if info['memory_bw'] is not None else "N/A"
            perf_index = info['perf_index'] if info['perf_index'] is not None else "N/A"
                
            print(f"| {cuda_idx:4d} | {pytorch_idx!s:6} | {nvitop_idx!s:9} | {pci_id:8} | {model:<{model_width-2}} | {vram:8.1f} | {matrix_perf!s:13} | {memory_bw!s:13} | {perf_index!s:11} |")
            
        print(separator_line)
        print("\nIMPORTANT: Pour utiliser un GPU spécifique, choisissez une des options suivantes:")
        print("  --cuda INDEX      : Utilise l'indice CUDA/NVML (colonne 1)")
        print("  --pytorch INDEX   : Utilise l'indice PyTorch (colonne 2) - RECOMMANDÉ pour le stress test")
        print("  --nvidia-smi INDEX: Utilise l'indice comme affiché dans nvidia-smi (colonne 3)")
        print("  --pci ID          : Utilise l'identifiant PCI (ex: 01:00.0)")
        
        # Afficher le meilleur GPU pour les calculs intensifs
        best_gpu = max(self.cuda_info.items(), key=lambda x: x[1]['perf_index'] if x[1]['perf_index'] is not None else 0)
        print(f"\nGPU le plus performant: CUDA {best_gpu[0]} ({best_gpu[1]['name']}) - Index de performance: {best_gpu[1]['perf_index']}")

    def __del__(self):
        # Assurez-vous que NVML est correctement fermé
        if self.nvml_working:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
            
    def log(self, message):
        """Écrit un message dans le log et l'affiche."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Écriture dans le fichier de log
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
            
    def get_gpu_info(self, cuda_idx):
        """Récupère les informations de température et d'utilisation du GPU."""
        info = {
            'gpu_temp': 'N/A',
            'mem_temp': 'N/A',
            'gpu_util': 'N/A',
            'mem_util': 'N/A'
        }
        
        if self.nvml_working:
            try:
                # IMPORTANT: Utiliser l'identifiant PCI pour récupérer le handle NVML
                # car les indices CUDA et les indices NVML peuvent être différents
                pci_id = self.cuda_info[cuda_idx]['pci_id']
                
                # Convertir le PCI ID en bytes pour NVML
                handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_id.encode('utf-8'))
                
                # Température GPU
                info['gpu_temp'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Température mémoire (VRAM)
                try:
                    info['mem_temp'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY)
                except:
                    info['mem_temp'] = 'N/S'  # Non supporté
                
                # Utilisation GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info['gpu_util'] = util.gpu
                info['mem_util'] = util.memory
                
            except Exception as e:
                self.log(f"Erreur lors de la lecture des infos pour GPU {cuda_idx} (PCI: {self.cuda_info[cuda_idx]['pci_id']}): {str(e)}")
                
        return info
    
    def monitor_gpus(self, cuda_indices):
        """Fonction de surveillance des températures des GPUs."""
        self.log(f"Démarrage de la surveillance des températures (intervalle: {self.monitoring_interval}s)")
        
        # Afficher les GPU surveillés
        gpu_str = []
        for cuda_idx in cuda_indices:
            gpu_info = self.cuda_info[cuda_idx]
            nvitop_idx = gpu_info['nvitop_idx'] if gpu_info['nvitop_idx'] is not None else "?"
            pytorch_idx = gpu_info['pytorch_idx'] if gpu_info['pytorch_idx'] is not None else "?"
            gpu_str.append(f"GPU {cuda_idx} [PyTorch: {pytorch_idx}, nvidia-smi: {nvitop_idx}, PCI: {gpu_info['pci_id'].split(':')[-2]}:{gpu_info['pci_id'].split(':')[-1]}]")
        self.log(f"Températures surveillées pour: {', '.join(gpu_str)}")
        
        # Initialisation des températures maximales
        for cuda_idx in cuda_indices:
            self.max_temps[cuda_idx] = {
                'gpu_temp': 0,
                'mem_temp': 0,
                'timestamp': None
            }
        
        count = 0
        while self.running:
            try:
                # Récupération de l'horodatage
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_line = timestamp
                
                # Ligne de statut pour l'affichage
                temps_status = []
                
                # Collecte des données pour tous les GPU
                for cuda_idx in range(self.num_gpus):
                    if cuda_idx in cuda_indices:
                        # Récupérer les infos fraîches à chaque itération
                        info = self.get_gpu_info(cuda_idx)
                        
                        # Mise à jour des températures maximales
                        if isinstance(info['gpu_temp'], int) and info['gpu_temp'] > self.max_temps[cuda_idx]['gpu_temp']:
                            self.max_temps[cuda_idx]['gpu_temp'] = info['gpu_temp']
                            self.max_temps[cuda_idx]['timestamp'] = timestamp
                        
                        if isinstance(info['mem_temp'], int) and info['mem_temp'] > self.max_temps[cuda_idx]['mem_temp']:
                            self.max_temps[cuda_idx]['mem_temp'] = info['mem_temp']
                        
                        # Construction de la ligne d'état pour ce GPU
                        nvitop_idx = self.cuda_info[cuda_idx]['nvitop_idx']
                        pytorch_idx = self.cuda_info[cuda_idx]['pytorch_idx'] if 'pytorch_idx' in self.cuda_info[cuda_idx] else "?"
                        nvitop_str = f"nvidia:{nvitop_idx}" if nvitop_idx is not None else "?"
                        pytorch_str = f"torch:{pytorch_idx}" if pytorch_idx is not None else "?"
                        pci_short = self.cuda_info[cuda_idx]['pci_id'].split(':')[-2] + ':' + self.cuda_info[cuda_idx]['pci_id'].split(':')[-1]
                        model_short = self.cuda_info[cuda_idx]['name'].split(' ')[-1]  # Just the model number
                        
                        temp_str = f"GPU {cuda_idx} [{pytorch_str}, {nvitop_str}, {model_short}, PCI:{pci_short}]: {info['gpu_temp']}°C"
                        if info['mem_temp'] != 'N/S':
                            temp_str += f"/VRAM: {info['mem_temp']}°C"
                        temp_str += f" (Util: {info['gpu_util']}%)"
                        temps_status.append(temp_str)
                        
                        # Données pour le CSV
                        gpu_temp = info['gpu_temp'] if info['gpu_temp'] != 'N/A' else ""
                        mem_temp = info['mem_temp'] if info['mem_temp'] != 'N/A' and info['mem_temp'] != 'N/S' else ""
                        gpu_util = info['gpu_util'] if info['gpu_util'] != 'N/A' else ""
                        mem_util = info['mem_util'] if info['mem_util'] != 'N/A' else ""
                    else:
                        # GPU non testé
                        gpu_temp = mem_temp = gpu_util = mem_util = ""
                    
                    csv_line += f",{gpu_temp},{mem_temp},{gpu_util},{mem_util}"
                
                # Affichage complet (une nouvelle ligne à chaque fois au lieu de \r)
                if count % 5 == 0:  # En-tête toutes les 5 lignes
                    print("\n" + "=" * 130)
                    print(f"[{timestamp}] TEMPÉRATURES DES GPU:")
                    print("=" * 130)
                
                # Afficher les températures sur une nouvelle ligne
                print(" | ".join(temps_status))
                
                # Écriture dans le CSV
                with open(self.temp_log_file, "a") as f:
                    f.write(csv_line + "\n")
                
                # Pause avant la prochaine mesure
                time.sleep(self.monitoring_interval)
                count += 1
                
            except Exception as e:
                self.log(f"Erreur dans le thread de surveillance: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
                # Continue malgré l'erreur
                time.sleep(self.monitoring_interval)
            
    def stress_gpu(self, cuda_idx):
        """Fonction qui exécute des calculs intensifs sur un GPU spécifique."""
        try:
            # IMPORTANT: Utiliser l'indice PyTorch correspondant pour set_device
            if cuda_idx in self.cuda_to_pytorch:
                pytorch_idx = self.cuda_to_pytorch[cuda_idx]
                self.log(f"Utilisation de l'indice PyTorch {pytorch_idx} pour le GPU CUDA {cuda_idx}")
                torch.cuda.set_device(pytorch_idx)
            else:
                # Si pas de correspondance trouvée, essayer directement (risqué)
                self.log(f"AVERTISSEMENT: Pas de correspondance PyTorch trouvée pour CUDA {cuda_idx}, utilisation directe (peut ne pas fonctionner)")
                torch.cuda.set_device(cuda_idx)
            
            gpu_info = self.cuda_info[cuda_idx]
            nvitop_idx = gpu_info['nvitop_idx'] if gpu_info['nvitop_idx'] is not None else "?"
            pytorch_idx = gpu_info['pytorch_idx'] if 'pytorch_idx' in gpu_info else "?"
            pci_id = gpu_info['pci_id']
            model_short = gpu_info['name'].split(' ')[-1]  # Just the model number
            
            self.log(f"Démarrage du stress test sur GPU {cuda_idx} [PyTorch: {pytorch_idx}, nvidia-smi: {nvitop_idx}, {model_short}, PCI: {pci_id}]")
            
            # Création des tenseurs pour le calcul
            dtype = torch.float16 if self.use_fp16 else torch.float32
            
            # Taille adaptative pour éviter les problèmes de mémoire
            try:
                self.log(f"Création des tenseurs de taille {self.tensor_size}x{self.tensor_size}...")
                tensor_a = torch.rand(self.tensor_size, self.tensor_size, device=f"cuda:{pytorch_idx if pytorch_idx != '?' else cuda_idx}", dtype=dtype)
                tensor_b = torch.rand(self.tensor_size, self.tensor_size, device=f"cuda:{pytorch_idx if pytorch_idx != '?' else cuda_idx}", dtype=dtype)
                self.log(f"Tenseurs créés avec succès, démarrage du calcul intensif...")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    reduced_size = self.tensor_size // 2
                    self.log(f"Erreur de mémoire, réduction de la taille à {reduced_size}x{reduced_size}...")
                    tensor_a = torch.rand(reduced_size, reduced_size, device=f"cuda:{pytorch_idx if pytorch_idx != '?' else cuda_idx}", dtype=dtype)
                    tensor_b = torch.rand(reduced_size, reduced_size, device=f"cuda:{pytorch_idx if pytorch_idx != '?' else cuda_idx}", dtype=dtype)
                else:
                    raise
            
            # Boucle de calcul intensif
            iteration = 0
            while self.running:
                # Phase de calcul intensif - effectuer plusieurs multiplications pour garantir une charge
                start_time = time.time()
                
                # Répéter plusieurs fois pour augmenter la charge
                for _ in range(3):
                    result = torch.matmul(tensor_a, tensor_b)
                    # Opération supplémentaire pour maximiser l'utilisation
                    result = torch.sigmoid(result)
                
                # Utiliser le résultat pour éviter l'optimisation
                dummy = result.sum().item()
                
                # Phase de pause pour contrôler l'utilisation
                compute_time = time.time() - start_time
                if self.utilization < 1.0:
                    sleep_time = compute_time * (1.0 - self.utilization) / self.utilization
                    time.sleep(max(0, sleep_time))
                
                # Afficher périodiquement l'état
                iteration += 1
                if iteration % 100 == 0:
                    self.log(f"GPU {cuda_idx}: {iteration} itérations effectuées")
                
                # Pour libérer la mémoire parfois
                if iteration % 500 == 0:
                    torch.cuda.empty_cache()
                
            self.log(f"Arrêt du stress test sur GPU {cuda_idx}")
        except Exception as e:
            self.log(f"Erreur dans le thread de stress pour GPU {cuda_idx}: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        
    def run(self, cuda_indices=None, pytorch_indices=None, nvitop_indices=None, pci_ids=None):
        """Démarre le stress test sur les GPUs spécifiés"""
        self.start_time = time.time()
        
        # Détermine quels GPU tester par leur indice CUDA
        cuda_indices_to_test = []
        
        if cuda_indices is not None:
            # Utiliser directement les indices CUDA spécifiés
            for cuda_idx in cuda_indices:
                if 0 <= cuda_idx < self.num_gpus:
                    cuda_indices_to_test.append(cuda_idx)
                else:
                    self.log(f"AVERTISSEMENT: Indice CUDA {cuda_idx} hors limites. Ignoré.")
        elif pytorch_indices is not None:
            # Convertir les indices PyTorch en indices CUDA
            for pytorch_idx in pytorch_indices:
                cuda_idx = self.pytorch_to_cuda.get(pytorch_idx)
                if cuda_idx is not None:
                    cuda_indices_to_test.append(cuda_idx)
                else:
                    self.log(f"AVERTISSEMENT: Indice PyTorch {pytorch_idx} non trouvé. Ignoré.")
        elif nvitop_indices is not None:
            # Convertir les indices nvitop en indices CUDA
            for nvitop_idx in nvitop_indices:
                cuda_idx = self.nvitop_to_cuda.get(nvitop_idx)
                if cuda_idx is not None:
                    cuda_indices_to_test.append(cuda_idx)
                else:
                    self.log(f"AVERTISSEMENT: Indice nvidia-smi {nvitop_idx} non trouvé. Ignoré.")
        elif pci_ids is not None:
            # Convertir les identifiants PCI en indices CUDA
            for pci_id in pci_ids:
                # Essayer d'abord le format complet
                cuda_idx = self.pci_to_cuda.get(pci_id)
                
                # Si non trouvé, essayer d'ajouter le préfixe
                if cuda_idx is None and not pci_id.startswith("0000"):
                    cuda_idx = self.pci_to_cuda.get(f"0000{pci_id}")
                
                # Si toujours non trouvé, essayer de trouver par la fin de l'identifiant
                if cuda_idx is None:
                    for full_pci, idx in self.pci_to_cuda.items():
                        if full_pci.endswith(pci_id):
                            cuda_idx = idx
                            break
                
                if cuda_idx is not None:
                    cuda_indices_to_test.append(cuda_idx)
                else:
                    self.log(f"AVERTISSEMENT: Identifiant PCI '{pci_id}' non trouvé. Ignoré.")
        else:
            # Tester tous les GPU
            cuda_indices_to_test = list(range(self.num_gpus))
            
        if not cuda_indices_to_test:
            self.log("ERREUR: Aucun GPU valide spécifié.")
            return
        
        # Log des informations de démarrage
        self.log(f"Démarrage du stress test sur {len(cuda_indices_to_test)} GPU(s):")
        for cuda_idx in cuda_indices_to_test:
            gpu_info = self.cuda_info[cuda_idx]
            nvitop_idx = gpu_info['nvitop_idx'] if gpu_info['nvitop_idx'] is not None else "?"
            pytorch_idx = gpu_info['pytorch_idx'] if 'pytorch_idx' in gpu_info else "?"
            perf_index = gpu_info['perf_index'] if gpu_info['perf_index'] is not None else "N/A"
            self.log(f"  GPU {cuda_idx} [PyTorch: {pytorch_idx}, nvidia-smi: {nvitop_idx}, PCI: {gpu_info['pci_id']}]: {gpu_info['name']} (Perf: {perf_index})")
            
        self.log(f"Taille de matrice: {self.tensor_size}x{self.tensor_size}, "
                f"Précision: {'FP16' if self.use_fp16 else 'FP32'}, "
                f"Utilisation cible: {int(self.utilization * 100)}%")
        self.log("Appuyez sur CTRL+C pour arrêter le test")
        
        # Démarrage du thread de surveillance si NVML est disponible
        if self.nvml_working:
            self.monitoring_thread = threading.Thread(target=self.monitor_gpus, args=(cuda_indices_to_test,))
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            # Brève pause pour s'assurer que le thread de surveillance démarre correctement
            time.sleep(0.5)
        else:
            self.log("AVERTISSEMENT: La surveillance des températures n'est pas disponible (pynvml manquant ou non fonctionnel)")
            
        # Création et démarrage d'un thread par GPU sélectionné
        for cuda_idx in cuda_indices_to_test:
            thread = threading.Thread(target=self.stress_gpu, args=(cuda_idx,))
            thread.daemon = True
            self.threads.append(thread)
            thread.start()
            # Petite pause entre chaque démarrage pour éviter la surcharge
            time.sleep(0.2)
            
        # Attente que tous les threads se terminent
        try:
            while any(thread.is_alive() for thread in self.threads):
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Arrête tous les threads de stress test."""
        self.running = False
        duration = time.time() - self.start_time
        self.log(f"Arrêt du stress test après {duration:.1f} secondes")
        
        # Attente que tous les threads se terminent
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        # Attente que le thread de surveillance se termine aussi
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
                
        # Affichage des températures maximales
        if self.nvml_working and self.max_temps:
            self.log("Températures maximales atteintes:")
            for cuda_idx, temps in self.max_temps.items():
                gpu_temp = temps['gpu_temp']
                mem_temp = temps['mem_temp'] if temps['mem_temp'] != 0 else "N/S"
                gpu_info = self.cuda_info[cuda_idx]
                nvitop_idx = gpu_info['nvitop_idx'] if gpu_info['nvitop_idx'] is not None else "?"
                pytorch_idx = gpu_info['pytorch_idx'] if 'pytorch_idx' in gpu_info else "?"
                self.log(f"  GPU {cuda_idx} [PyTorch: {pytorch_idx}, nvidia-smi: {nvitop_idx}, PCI: {gpu_info['pci_id']}]: {gpu_temp}°C, VRAM: {mem_temp}°C")
        
        self.log(f"Stress test terminé. Logs sauvegardés dans:")
        self.log(f"  - {self.log_file} (messages)")
        if self.nvml_working:
            self.log(f"  - {self.temp_log_file} (températures)")
        
    def signal_handler(self, sig, frame):
        """Gestionnaire de signal pour CTRL+C."""
        self.log("Signal d'interruption reçu. Arrêt en cours...")
        self.stop()
        sys.exit(0)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test GPU avec CUDA")
    parser.add_argument("--size", type=int, default=8000, 
                       help="Taille des matrices (default: 8000)")
    parser.add_argument("--fp16", action="store_true", 
                       help="Utiliser FP16 au lieu de FP32 pour les calculs")
    parser.add_argument("--util", type=int, default=95, 
                       help="Pourcentage d'utilisation cible (10-100, default: 95)")
    
    # Groupes mutuellement exclusifs pour spécifier les GPU
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--cuda", type=int, nargs='+',
                       help="ID(s) CUDA/NVML des GPU à tester (ex: 0 1 3)")
    gpu_group.add_argument("--pytorch", type=int, nargs='+',
                       help="ID(s) PyTorch des GPU à tester - RECOMMANDÉ (ex: 0 1 3)")
    gpu_group.add_argument("--nvidia-smi", type=int, nargs='+',
                       help="ID(s) des GPU comme affichés dans nvidia-smi (ex: 0 4)")
    gpu_group.add_argument("--pci", type=str, nargs='+',
                       help="ID(s) de bus PCI des GPU à tester (ex: 01:00.0)")
    
    parser.add_argument("--list", action="store_true",
                       help="Afficher la liste des GPU disponibles et quitter")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Intervalle de surveillance des températures en secondes (default: 1.0)")
    parser.add_argument("--detection-duration", type=float, default=4.0,
                       help="Durée du test pour la détection d'index PyTorch en secondes (default: 4.0)")
    parser.add_argument("--detailed-benchmark", action="store_true",
                       help="Afficher des informations détaillées sur les benchmarks")
    
    # Nouvelles options pour le cache de mapping
    parser.add_argument("--force-detection", action="store_true",
                       help="Forcer une nouvelle détection des index PyTorch (ignore le cache)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Ne pas utiliser ni mettre à jour le cache de mapping")
    parser.add_argument("--mapping-file", type=str,
                       help="Spécifier un fichier de mapping personnalisé (default: gpu_mapping_cache.json)")
    
    args = parser.parse_args()
    
    # Instanciation du test
    stress_test = GPUStressTest(
        tensor_size=args.size,
        use_fp16=args.fp16,
        utilization=args.util,
        monitoring_interval=args.interval,
        detection_duration=args.detection_duration,
        use_cached_mapping=not args.no_cache,
        force_detection=args.force_detection,
        mapping_file=args.mapping_file
    )
    
    # Affichage de la liste des GPU si demandé
    if args.list:
        sys.exit(0)
    
    # Exécution du test sur les GPU spécifiés
    stress_test.run(
        cuda_indices=args.cuda,
        pytorch_indices=args.pytorch,
        nvitop_indices=args.nvidia_smi,
        pci_ids=args.pci
    )