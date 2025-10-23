r"""
Activity Filter - Filtra chunks de video por atividade

Detecta movimento e presenca de pessoas para identificar chunks relevantes,
descartando gravacoes vazias ou sem atividade significativa.

Tecnicas:
- Frame differencing para deteccao rapida de movimento
- YOLO person detection para confirmacao de presenca humana
- Sampling inteligente para otimizar performance

Resultado esperado: Reducao de 60-80% de video sem atividade
"""

import cv2
import numpy as np
import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime


class ActivityFilter:
    """
    Filtra chunks de video sem atividade relevante
    
    Processo em duas etapas:
    1. Check rapido: Frame differencing para detectar movimento geral
    2. Check preciso: YOLO person detection para confirmar presenca humana
    
    Attributes:
        motion_threshold: Percentual minimo de pixels com movimento (0.0-1.0)
        min_person_frames: Minimo de frames com pessoa detectada
        person_detection_model: Path do modelo YOLO para deteccao
        motion_sample_rate: Processar 1 a cada N frames (fase movimento)
        person_sample_rate: Processar 1 a cada N frames (fase pessoa)
    """
    
    def __init__(
        self,
        motion_threshold: float = 0.02,
        min_person_frames: int = 30,
        person_detection_model: str = 'yolo11n.pt',
        person_conf_threshold: float = 0.5,
        motion_sample_rate: int = 10,
        person_sample_rate: int = 15,
        min_bbox_area: int = 2000,
        max_bbox_area: int = 500000,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 4.0
    ):
        """
        Inicializa ActivityFilter
        
        Args:
            motion_threshold: % de pixels com movimento (0.02 = 2%)
            min_person_frames: Minimo de frames com pessoa
            person_detection_model: Modelo YOLO leve para deteccao rapida
            person_conf_threshold: Confidence minimo para deteccao de pessoa
            motion_sample_rate: Sample rate para deteccao de movimento
            person_sample_rate: Sample rate para deteccao de pessoa
            min_bbox_area: Area minima da bbox (pixels)
            max_bbox_area: Area maxima da bbox (pixels)
            min_aspect_ratio: Aspect ratio minimo (altura/largura)
            max_aspect_ratio: Aspect ratio maximo (altura/largura)
        """
        self.motion_threshold = motion_threshold
        self.min_person_frames = min_person_frames
        self.person_conf_threshold = person_conf_threshold
        self.motion_sample_rate = motion_sample_rate
        self.person_sample_rate = person_sample_rate
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Importar GPUManager
        try:
            from ..utils.gpu_manager import GPUManager
            self.gpu_manager = GPUManager()
        except ImportError:
            self.logger.warning("GPUManager nao encontrado - usando deteccao automatica de GPU")
            self.gpu_manager = None
        
        # Antes de carregar YOLO, aplicar fallback para torchvision.ops.nms caso o operador nativo nao exista
        try:
            import torch
            import torchvision
            from torchvision import ops as tv_ops
            try:
                _ = tv_ops.nms(torch.zeros((1, 4)), torch.zeros((1,)), 0.5)
            except Exception:
                def _nms_fallback(boxes: 'torch.Tensor', scores: 'torch.Tensor', iou_threshold: float):
                    # implementacao simples de nms em pytorch puro
                    if boxes.numel() == 0:
                        return torch.empty((0,), dtype=torch.long, device=boxes.device)
                    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
                    order = scores.sort(descending=True).indices
                    keep = []
                    while order.numel() > 0:
                        i = int(order[0])
                        keep.append(i)
                        if order.numel() == 1:
                            break
                        xx1 = torch.maximum(x1[i], x1[order[1:]])
                        yy1 = torch.maximum(y1[i], y1[order[1:]])
                        xx2 = torch.minimum(x2[i], x2[order[1:]])
                        yy2 = torch.minimum(y2[i], y2[order[1:]])
                        w = (xx2 - xx1).clamp(min=0)
                        h = (yy2 - yy1).clamp(min=0)
                        inter = w * h
                        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
                        mask = iou <= iou_threshold
                        order = order[1:][mask]
                    return torch.tensor(keep, device=boxes.device, dtype=torch.long)
                tv_ops.nms = _nms_fallback  # monkey patch
                self.logger.warning("torchvision.ops.nms nativo indisponivel, usando fallback pytorch puro")
        except Exception:
            # Se torchvision nao estiver disponivel, seguir (ultralytics pode usar nms proprio)
            pass

        # Carregar modelo YOLO leve com configuracao GPU
        try:
            from ultralytics import YOLO
            
            # Log de status GPU
            if self.gpu_manager:
                self.gpu_manager.log_component_init("ActivityFilter", "YOLO")
                device_config = self.gpu_manager.get_yolo_device_config()
            else:
                device_config = None
            
            self.detector = YOLO(person_detection_model)
            
            # Configurar device explicitamente se disponivel
            if device_config and hasattr(self.detector, 'to'):
                self.detector.to(device_config)
                self.logger.info(f"YOLO configurado para device: {device_config}")
            
            self.logger.info(f"Modelo YOLO carregado: {person_detection_model}")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo YOLO: {e}")
            raise
    
    def filter_inactive_chunks(
        self,
        chunks_metadata: List[Dict],
        output_dir: Optional[str] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Filtra lista de chunks removendo os sem atividade
        
        Args:
            chunks_metadata: Lista de dicts com metadata dos chunks (do VideoChunker)
            output_dir: Diretorio para salvar relatorio (opcional)
        
        Returns:
            Tupla com:
            - Lista de chunks com atividade detectada
            - Dict com estatisticas do processamento
        """
        self.logger.info(f"Iniciando filtragem de {len(chunks_metadata)} chunks...")
        
        active_chunks = []
        stats = {
            'total_chunks': len(chunks_metadata),
            'active_chunks': 0,
            'inactive_chunks': 0,
            'motion_rejected': 0,
            'person_rejected': 0,
            'processing_time_seconds': 0,
            'start_time': datetime.now().isoformat()
        }
        
        start_time = datetime.now()
        
        for idx, chunk_info in enumerate(chunks_metadata):
            chunk_path = chunk_info['filepath']
            chunk_id = chunk_info.get('chunk_id', f'chunk_{idx:04d}')
            
            self.logger.info(f"[{idx+1}/{len(chunks_metadata)}] Processando {chunk_id}...")
            
            # Verificar existencia do arquivo
            if not os.path.exists(chunk_path):
                self.logger.warning(f"Chunk nao encontrado: {chunk_path}")
                continue
            
            # Etapa 1: Verificar movimento geral (rapido)
            has_motion = self._detect_motion(chunk_path)
            
            if not has_motion:
                self.logger.info(f"  └─ Sem movimento significativo (< {self.motion_threshold*100:.1f}%)")
                stats['motion_rejected'] += 1
                stats['inactive_chunks'] += 1
                continue
            
            # Etapa 2: Confirmar presenca de pessoas (mais lento)
            person_frames, total_sampled = self._count_person_frames(chunk_path)
            
            if person_frames < self.min_person_frames:
                self.logger.info(
                    f"  └─ Poucas pessoas detectadas ({person_frames} frames < {self.min_person_frames})"
                )
                stats['person_rejected'] += 1
                stats['inactive_chunks'] += 1
                continue
            
            # Chunk aprovado!
            activity_score = person_frames / total_sampled if total_sampled > 0 else 0
            
            chunk_info['person_frames'] = person_frames
            chunk_info['total_sampled_frames'] = total_sampled
            chunk_info['activity_score'] = activity_score
            chunk_info['filter_timestamp'] = datetime.now().isoformat()
            
            active_chunks.append(chunk_info)
            stats['active_chunks'] += 1
            
            self.logger.info(
                f"  └─ ATIVO! ({person_frames} pessoas em {total_sampled} frames, "
                f"score={activity_score:.2%})"
            )
        
        # Calcular tempo total
        end_time = datetime.now()
        stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
        stats['end_time'] = end_time.isoformat()
        
        # Log resumo
        retention_rate = stats['active_chunks'] / stats['total_chunks'] * 100
        self.logger.info(
            f"\n=== RESUMO DA FILTRAGEM ===\n"
            f"Total de chunks: {stats['total_chunks']}\n"
            f"Chunks ativos: {stats['active_chunks']} ({retention_rate:.1f}%)\n"
            f"Chunks inativos: {stats['inactive_chunks']} ({100-retention_rate:.1f}%)\n"
            f"  - Rejeitados por movimento: {stats['motion_rejected']}\n"
            f"  - Rejeitados por pessoas: {stats['person_rejected']}\n"
            f"Tempo de processamento: {stats['processing_time_seconds']:.1f}s\n"
        )
        
        # Salvar relatorio se output_dir fornecido
        if output_dir:
            self._save_report(active_chunks, stats, output_dir)
        
        return active_chunks, stats
    
    def _detect_motion(self, video_path: str) -> bool:
        """
        Detecta movimento via frame differencing
        
        Estrategia: Compara frames consecutivos e calcula % de pixels com diferenca.
        Se qualquer par de frames tiver movimento > threshold, retorna True.
        
        Args:
            video_path: Caminho do video chunk
        
        Returns:
            True se movimento detectado, False caso contrario
        """
        cap = cv2.VideoCapture(video_path)
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            self.logger.warning(f"Nao foi possivel ler primeiro frame de {video_path}")
            return False
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_detected = False
        frames_checked = 0
        
        while True:
            # Pular frames para acelerar (sample rate)
            for _ in range(self.motion_sample_rate - 1):
                cap.grab()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calcular diferenca entre frames
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # % de pixels com movimento
            motion_pixels = np.count_nonzero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            motion_ratio = motion_pixels / total_pixels
            
            frames_checked += 1
            
            if motion_ratio > self.motion_threshold:
                motion_detected = True
                break
            
            prev_gray = gray
        
        cap.release()
        
        self.logger.debug(
            f"Motion check: {frames_checked} frames verificados, "
            f"movimento={'SIM' if motion_detected else 'NAO'}"
        )
        
        return motion_detected
    
    def _count_person_frames(self, video_path: str) -> Tuple[int, int]:
        """
        Conta frames com pessoa detectada usando YOLO
        
        Args:
            video_path: Caminho do video chunk
        
        Returns:
            Tupla (person_frames, total_sampled)
            - person_frames: Numero de frames com pessoa detectada
            - total_sampled: Total de frames processados (para calcular score)
        """
        cap = cv2.VideoCapture(video_path)
        
        person_frames = 0
        total_sampled = 0
        
        while True:
            # Pular frames para acelerar (sample rate)
            for _ in range(self.person_sample_rate - 1):
                cap.grab()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            total_sampled += 1
            
            # Detectar pessoas (classe 0 no COCO dataset)
            results = self.detector.predict(
                frame,
                verbose=False,
                conf=self.person_conf_threshold,  # Usar parametro configuravel
                classes=[0]  # Apenas classe "person"
            )
            
            # Verificar se alguma pessoa foi detectada COM VALIDACAO DE BBOX
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Validar qualidade das bboxes
                    valid_detection = False
                    
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Calcular area e aspect ratio
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect_ratio = height / width if width > 0 else 0
                        
                        # Filtros de qualidade
                        if (self.min_bbox_area <= area <= self.max_bbox_area and
                            self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                            valid_detection = True
                            break
                    
                    if valid_detection:
                        person_frames += 1
                        break  # Ja encontrou pessoa valida neste frame, proximo
        
        cap.release()
        
        self.logger.debug(
            f"Person check: {person_frames} pessoas em {total_sampled} frames amostrados"
        )
        
        return person_frames, total_sampled
    
    def _save_report(
        self,
        active_chunks: List[Dict],
        stats: Dict,
        output_dir: str
    ):
        """
        Salva relatorio de filtragem em JSON
        
        Args:
            active_chunks: Lista de chunks ativos
            stats: Estatisticas do processamento
            output_dir: Diretorio de saida
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'statistics': stats,
            'active_chunks': active_chunks,
            'filter_config': {
                'motion_threshold': self.motion_threshold,
                'min_person_frames': self.min_person_frames,
                'motion_sample_rate': self.motion_sample_rate,
                'person_sample_rate': self.person_sample_rate
            }
        }
        
        report_path = os.path.join(output_dir, 'active_chunks_report.json')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Relatorio salvo em: {report_path}")
    
    @staticmethod
    def load_report(report_path: str) -> Tuple[List[Dict], Dict]:
        """
        Carrega relatorio de filtragem previamente salvo
        
        Args:
            report_path: Caminho do arquivo JSON
        
        Returns:
            Tupla (active_chunks, stats)
        """
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        return report['active_chunks'], report['statistics']


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Exemplo: Filtrar chunks de um diretorio
    from video_chunker import VideoChunker
    
    # 1. Carregar metadata de chunks existentes
    # Por padrao, nao habilitamos GPU aqui (deixe o caller passar use_gpu se necessario)
    chunker = VideoChunker()
    chunks_metadata = chunker.load_chunks_index('data/chunks/chunks_index.json')
    
    # 2. Filtrar chunks inativos
    filter = ActivityFilter(
        motion_threshold=0.02,      # 2% de movimento
        min_person_frames=30,       # Minimo 30 frames com pessoa
        person_detection_model='yolo11n.pt'
    )
    
    active_chunks, stats = filter.filter_inactive_chunks(
        chunks_metadata,
        output_dir='data/active_chunks'
    )
    
    # 3. Resultados
    print(f"\n=== RESULTADOS ===")
    print(f"Chunks ativos: {len(active_chunks)}")
    print(f"Reducao: {100 - (len(active_chunks)/len(chunks_metadata)*100):.1f}%")
    print(f"Tempo total: {stats['processing_time_seconds']:.1f}s")


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
