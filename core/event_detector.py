r"""
Event Detector - Detecta e agrupa eventos relevantes com tracking

Processa chunks ativos detectando pessoas e rastreando movimentos ao longo
do tempo usando ByteTrack. Agrupa deteccoes consecutivas em eventos unicos.

Tecnicas:
- YOLO11m para deteccao precisa de pessoas
- ByteTrack para tracking persistente entre frames
- Agrupamento por track_id para eventos temporais
- Filtragem por duracao minima

Output: Eventos estruturados prontos para classificacao
"""

import cv2
import json
import os
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class EventDetector:
    """
    Detecta eventos relevantes usando person detection + tracking
    
    Pipeline:
    1. Processar cada chunk ativo com YOLO + tracking
    2. Agrupar deteccoes por track_id
    3. Filtrar eventos muito curtos (< min_duration)
    4. Gerar metadata completo de cada evento
    
    Attributes:
        detector_model: Path do modelo YOLO para deteccao
        tracker_config: Config do tracker (bytetrack.yaml)
        confidence_threshold: Confidence minimo para deteccao
        iou_threshold: IOU minimo para matching
        min_duration_seconds: Duracao minima do evento
    """
    
    def __init__(
        self,
        detector_model: str = 'yolo11m.pt',
        tracker_config: str = 'bytetrack.yaml',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        min_duration_seconds: float = 1.0,
        sample_rate: int = 1,
        min_bbox_area: int = 2000,
        max_bbox_area: int = 500000,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 4.0,
        min_track_length: int = 15,
        min_track_confidence_avg: float = 0.55,
        require_motion_for_event: bool = True,
        min_track_movement_pixels: float = 12.0
    ):
        """
        Inicializa EventDetector
        
        Args:
            detector_model: Modelo YOLO medio/pesado para precisao
            tracker_config: Arquivo YAML do tracker
            confidence_threshold: Conf minimo (0.0-1.0)
            iou_threshold: IOU minimo para NMS
            min_duration_seconds: Duracao minima de evento valido
            sample_rate: Processar 1 a cada N frames (1=todos, 3=10fps)
            min_bbox_area: Area minima da bbox (pixels)
            max_bbox_area: Area maxima da bbox (pixels)
            min_aspect_ratio: Aspect ratio minimo (altura/largura)
            max_aspect_ratio: Aspect ratio maximo (altura/largura)
            min_track_length: Minimo de deteccoes por track
        """
        self.detector_model = detector_model
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_duration_seconds = min_duration_seconds
        self.sample_rate = sample_rate
        self.min_bbox_area = min_bbox_area
        self.max_bbox_area = max_bbox_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_track_length = min_track_length
        self.min_track_confidence_avg = min_track_confidence_avg
        self.require_motion_for_event = require_motion_for_event
        self.min_track_movement_pixels = min_track_movement_pixels
      
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Importar GPUManager
        try:
            from ..utils.gpu_manager import GPUManager
            self.gpu_manager = GPUManager()
        except ImportError:
            self.logger.warning("GPUManager nao encontrado - usando deteccao automatica de GPU")
            self.gpu_manager = None
        
        # Observacao: evitar importar torchvision aqui para nao acionar erro de operador ausente

        # Carregar modelo YOLO com configuracao GPU
        try:
            from ultralytics import YOLO
            print(f"[EventDetector] Carregando modelo YOLO: {detector_model}...", flush=True)
            
            # Log de status GPU
            if self.gpu_manager:
                self.gpu_manager.log_component_init("EventDetector", "YOLO")
                device_config = self.gpu_manager.get_yolo_device_config()
            else:
                device_config = None
            
            self.detector = YOLO(detector_model)
            
            # Configurar device explicitamente se disponivel
            if device_config and hasattr(self.detector, 'to'):
                self.detector.to(device_config)
                print(f"[EventDetector] YOLO configurado para device: {device_config}", flush=True)
            
            print(f"[EventDetector] Modelo YOLO carregado com sucesso!", flush=True)
            self.logger.info(f"Modelo YOLO carregado: {detector_model}")
        except Exception as e:
            print(f"[EventDetector] ERRO ao carregar YOLO: {e}", flush=True)
            self.logger.error(f"Erro ao carregar modelo YOLO: {e}")
            raise
    
    def detect_events_batch(
        self,
        active_chunks: List[Dict],
        output_dir: Optional[str] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Detecta eventos em uma lista de chunks ativos
        
        Args:
            active_chunks: Lista de chunks com atividade (do ActivityFilter)
            output_dir: Diretorio para salvar eventos (opcional)
        
        Returns:
            Tupla com:
            - Lista de eventos detectados
            - Dict com estatisticas do processamento
        """
        print(f"\n[EventDetector] Iniciando deteccao em {len(active_chunks)} chunks ativos...", flush=True)
        self.logger.info(f"Iniciando deteccao em {len(active_chunks)} chunks ativos...")
        self.logger.info(f"Modelo: {self.detector_model} | Tracker: {self.tracker_config}")
        self.logger.info(f"Confidence: {self.confidence_threshold} | Min duration: {self.min_duration_seconds}s")
        self.logger.info("")
        print(f"[EventDetector] Configuracoes: conf={self.confidence_threshold}, min_dur={self.min_duration_seconds}s", flush=True)
        
        all_events = []
        stats = {
            'total_chunks': len(active_chunks),
            'total_events': 0,
            'total_tracks': 0,
            'events_by_duration': {
                '<1s': 0,
                '1-5s': 0,
                '5-15s': 0,
                '15-30s': 0,
                '>30s': 0
            },
            'processing_time_seconds': 0,
            'start_time': datetime.now().isoformat()
        }
        
        start_time = datetime.now()
        
        # Barra de progresso
        total_chunks = len(active_chunks)
        progress_width = 40
        
        for idx, chunk_info in enumerate(active_chunks):
            chunk_path = chunk_info['filepath']
            chunk_id = chunk_info.get('chunk_id', f'chunk_{idx:04d}')
            
            # Progresso
            progress = (idx + 1) / total_chunks
            filled = int(progress_width * progress)
            bar = '█' * filled + '░' * (progress_width - filled)
            percent = progress * 100
            
            print(f"\n[{idx+1}/{total_chunks}] {bar} {percent:.1f}% | {chunk_id}", flush=True)
            
            self.logger.info(
                f"[{idx+1}/{total_chunks}] {bar} {percent:.1f}% | "
                f"Processando {chunk_id}..."
            )
            
            print(f"  -> Iniciando tracking...", flush=True)
            
            # Detectar eventos no chunk
            events = self.detect_events_in_chunk(chunk_path, chunk_info)
            
            # Adicionar contexto do chunk
            for event in events:
                event['chunk_id'] = chunk_id
                event['chunk_filepath'] = chunk_path
                if 'start_timestamp' in chunk_info:
                    event['chunk_start_timestamp'] = chunk_info['start_timestamp']
            
            all_events.extend(events)
            
            # Tempo decorrido e estimativa
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time_per_chunk = elapsed / (idx + 1)
            remaining_chunks = total_chunks - (idx + 1)
            eta_seconds = avg_time_per_chunk * remaining_chunks
            eta_minutes = eta_seconds / 60
            
            self.logger.info(
                f"  └─ {len(events)} eventos detectados | "
                f"Tempo: {avg_time_per_chunk:.1f}s/chunk | "
                f"ETA: {eta_minutes:.1f} min"
            )
        
        # Estatisticas finais
        stats['total_events'] = len(all_events)
        stats['total_tracks'] = len(set(e['track_id'] for e in all_events))
        
        # Distribuicao por duracao
        for event in all_events:
            duration = event['duration_seconds']
            if duration < 1:
                stats['events_by_duration']['<1s'] += 1
            elif duration < 5:
                stats['events_by_duration']['1-5s'] += 1
            elif duration < 15:
                stats['events_by_duration']['5-15s'] += 1
            elif duration < 30:
                stats['events_by_duration']['15-30s'] += 1
            else:
                stats['events_by_duration']['>30s'] += 1
        
        end_time = datetime.now()
        stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
        stats['end_time'] = end_time.isoformat()
        
        # Log resumo
        self.logger.info(
            f"\n=== RESUMO DA DETECCAO ===\n"
            f"Chunks processados: {stats['total_chunks']}\n"
            f"Total de eventos: {stats['total_events']}\n"
            f"Tracks unicos: {stats['total_tracks']}\n"
            f"Duracao dos eventos:\n"
            f"  - <1s: {stats['events_by_duration']['<1s']}\n"
            f"  - 1-5s: {stats['events_by_duration']['1-5s']}\n"
            f"  - 5-15s: {stats['events_by_duration']['5-15s']}\n"
            f"  - 15-30s: {stats['events_by_duration']['15-30s']}\n"
            f"  - >30s: {stats['events_by_duration']['>30s']}\n"
            f"Tempo de processamento: {stats['processing_time_seconds']:.1f}s\n"
        )
        
        # Salvar eventos se output_dir fornecido
        if output_dir:
            self._save_events(all_events, stats, output_dir)
        
        return all_events, stats
    
    def detect_events_in_chunk(
        self,
        chunk_path: str,
        chunk_info: Dict
    ) -> List[Dict]:
        """
        Detecta eventos em um unico chunk
        
        Args:
            chunk_path: Caminho do video chunk
            chunk_info: Metadata do chunk (do VideoChunker)
        
        Returns:
            Lista de eventos detectados no chunk
        """
        self.logger.info(f"  -> Iniciando tracking no chunk...")
        
        # Processar video com tracking (stream=True para evitar acumulo de RAM)
        results = self.detector.track(
            source=chunk_path,
            persist=True,  # Manter IDs entre frames
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_config,
            classes=[0],  # Apenas classe "person"
            verbose=False,
            stream=True  # Generator mode para economizar memoria
        )
        
        # Agrupar deteccoes por track_id
        tracks = defaultdict(list)
        frame_count = 0
        frames_processed = 0
        frame_w = None
        frame_h = None
        
        print(f"  -> Processando frames do chunk (sample_rate={self.sample_rate})...", flush=True)
        
        for frame_idx, result in enumerate(results):
            frame_count += 1
            
            # Pular frames de acordo com sample_rate (otimizacao)
            if frame_count % self.sample_rate != 0:
                continue
            
            frames_processed += 1
            
            # Log a cada 100 frames processados
            if frames_processed % 100 == 0:
                print(f"     Frame {frames_processed} processado (total: {frame_count})...", flush=True)
            
            # Capturar tamanho do frame para normalizacao de zonas
            if frame_w is None and hasattr(result, 'orig_shape') and result.orig_shape is not None:
                try:
                    frame_h, frame_w = int(result.orig_shape[0]), int(result.orig_shape[1])
                except Exception:
                    frame_w, frame_h = None, None

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                # Normalizar saidas para listas python de forma robusta
                ids_attr = getattr(boxes, 'id', None)
                conf_attr = getattr(boxes, 'conf', None)
                xyxy_attr = getattr(boxes, 'xyxy', None)

                def _to_list(arr, cast=float):
                    try:
                        import numpy as _np  # type: ignore
                    except Exception:
                        _np = None
                    try:
                        a = arr
                        if hasattr(a, 'cpu'):
                            a = a.cpu()
                        if hasattr(a, 'numpy'):
                            a = a.numpy()
                        elif _np is not None and isinstance(a, _np.ndarray):
                            pass
                        else:
                            return list(a) if isinstance(a, (list, tuple)) else []
                        lst = a.tolist()
                        # Achatar listas aninhadas simples
                        if lst and isinstance(lst[0], list) and len(lst[0]) == 1:
                            lst = [x[0] for x in lst]
                        return [cast(x) for x in lst]
                    except Exception:
                        return []

                track_ids = _to_list(ids_attr, cast=int)
                confidences = _to_list(conf_attr, cast=float)
                xyxy = _to_list(xyxy_attr, cast=float)
                
                for track_id, conf, bbox in zip(track_ids, confidences, xyxy):
                    x1, y1, x2, y2 = bbox
                    
                    # Validar qualidade da bbox
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0
                    
                    # Filtros de qualidade
                    if not (self.min_bbox_area <= area <= self.max_bbox_area):
                        continue  # Bbox muito pequena ou muito grande
                    
                    if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                        continue  # Aspect ratio invalido
                    
                    # Bbox valida, adicionar ao track
                    tracks[track_id].append({
                        'frame': frame_idx,
                        'bbox': bbox,  # [x1, y1, x2, y2]
                        'confidence': conf
                    })
        
        print(f"  -> {frame_count} frames totais, {frames_processed} processados, {len(tracks)} tracks encontrados", flush=True)
        self.logger.info(f"  -> {len(tracks)} tracks encontrados, filtrando eventos...")
        
        # Converter tracks em eventos
        fps = chunk_info.get('fps', 30.0)  # Default 30 FPS
        events = []
        
        # Contadores de filtragem para diagnostico
        filter_stats = {
            'total_tracks': len(tracks),
            'rejected_track_length': 0,
            'rejected_duration': 0,
            'rejected_confidence': 0,
            'rejected_movement': 0,
            'accepted': 0
        }
        
        for track_id, detections in tracks.items():
            # Filtrar eventos muito curtos (minimo de deteccoes)
            if len(detections) < self.min_track_length:
                filter_stats['rejected_track_length'] += 1
                self.logger.debug(f"    Track {track_id}: rejeitado por track_length ({len(detections)} < {self.min_track_length})")
                continue
            
            start_frame = detections[0]['frame']
            end_frame = detections[-1]['frame']
            duration = (end_frame - start_frame) / fps
            
            if duration < self.min_duration_seconds:
                filter_stats['rejected_duration'] += 1
                self.logger.debug(f"    Track {track_id}: rejeitado por duracao ({duration:.2f}s < {self.min_duration_seconds}s)")
                continue
            
            # Calcular estatisticas
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            if avg_conf < self.min_track_confidence_avg:
                filter_stats['rejected_confidence'] += 1
                self.logger.debug(f"    Track {track_id}: rejeitado por confianca ({avg_conf:.2f} < {self.min_track_confidence_avg})")
                continue
            
            # Calcular movimento (distancia entre primeira e ultima bbox)
            bbox_start = detections[0]['bbox']
            bbox_end = detections[-1]['bbox']
            
            center_start = [
                (bbox_start[0] + bbox_start[2]) / 2,
                (bbox_start[1] + bbox_start[3]) / 2
            ]
            center_end = [
                (bbox_end[0] + bbox_end[2]) / 2,
                (bbox_end[1] + bbox_end[3]) / 2
            ]
            
            movement_distance = (
                (center_end[0] - center_start[0])**2 +
                (center_end[1] - center_start[1])**2
            )**0.5
            if self.require_motion_for_event and movement_distance < self.min_track_movement_pixels:
                # Poco movimento entre inicio e fim do track (possivel objeto estatico)
                filter_stats['rejected_movement'] += 1
                self.logger.debug(f"    Track {track_id}: rejeitado por movimento ({movement_distance:.1f}px < {self.min_track_movement_pixels}px)")
                continue
            
            # Track passou por todos os filtros
            filter_stats['accepted'] += 1
            
            # Criar evento
            event = {
                'event_id': f"event_{track_id:04d}",
                'track_id': track_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_seconds': duration,
                'frame_count': len(detections),
                'confidence_avg': avg_conf,
                'confidence_min': min(d['confidence'] for d in detections),
                'confidence_max': max(d['confidence'] for d in detections),
                'movement_distance': movement_distance,
                'bbox_sequence': [d['bbox'] for d in detections],
                'detection_timestamps': datetime.now().isoformat()
            }
            
            events.append(event)
        
        # Log estatisticas de filtragem
        print(f"  -> Filtragem: {filter_stats['total_tracks']} tracks | "
              f"Rejeitados: length={filter_stats['rejected_track_length']}, "
              f"duration={filter_stats['rejected_duration']}, "
              f"conf={filter_stats['rejected_confidence']}, "
              f"motion={filter_stats['rejected_movement']} | "
              f"ACEITOS: {filter_stats['accepted']}", flush=True)
        
        self.logger.info(
            f"  -> Estatisticas de filtragem: "
            f"total={filter_stats['total_tracks']}, "
            f"rej_length={filter_stats['rejected_track_length']}, "
            f"rej_duration={filter_stats['rejected_duration']}, "
            f"rej_conf={filter_stats['rejected_confidence']}, "
            f"rej_motion={filter_stats['rejected_movement']}, "
            f"aceitos={filter_stats['accepted']}"
        )
        
        return events
    
    
    def _save_events(
        self,
        events: List[Dict],
        stats: Dict,
        output_dir: str
    ):
        """
        Salva eventos detectados em JSON
        
        Args:
            events: Lista de eventos
            stats: Estatisticas do processamento
            output_dir: Diretorio de saida
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'statistics': stats,
            'events': events,
            'detector_config': {
                'model': self.detector_model,
                'tracker': self.tracker_config,
                'confidence_threshold': self.confidence_threshold,
                'iou_threshold': self.iou_threshold,
                'min_duration_seconds': self.min_duration_seconds
            }
        }
        
        report_path = os.path.join(output_dir, 'events_summary.json')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Eventos salvos em: {report_path}")
    
    @staticmethod
    def load_events(events_path: str) -> Tuple[List[Dict], Dict]:
        """
        Carrega eventos previamente detectados
        
        Args:
            events_path: Caminho do arquivo JSON
        
        Returns:
            Tupla (events, stats)
        """
        with open(events_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        return report['events'], report['statistics']


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Exemplo: Detectar eventos em chunks ativos
    from activity_filter import ActivityFilter
    
    # 1. Carregar chunks ativos
    active_chunks, _ = ActivityFilter.load_report('data/active_chunks/active_chunks_report.json')
    
    # 2. Detectar eventos
    detector = EventDetector(
        detector_model='yolo11m.pt',
        tracker_config='bytetrack.yaml',
        confidence_threshold=0.25,
        min_duration_seconds=0.5
    )
    
    events, stats = detector.detect_events_batch(
        active_chunks,
        output_dir='data/events'
    )
    
    # 3. Resultados
    print(f"\n=== RESULTADOS ===")
    print(f"Total de eventos: {len(events)}")
    print(f"Tracks unicos: {stats['total_tracks']}")
    print(f"Tempo total: {stats['processing_time_seconds']:.1f}s")
    print(f"Eventos/hora: {len(events) / (stats['processing_time_seconds']/3600):.0f}")


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
