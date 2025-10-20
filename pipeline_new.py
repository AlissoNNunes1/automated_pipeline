r"""
Pipeline Automatizado: Processa todos os videos .dav automaticamente

Sistema totalmente automatico que:
1. Detecta videos .dav em videos_full/
2. Converte para MP4
3. Divide em chunks
4. Filtra chunks sem atividade
5. Detecta eventos
6. Gera propostas de anotacao
7. Gerencia estado para evitar reprocessamento

Uso:
    python pipeline.py  # Processa tudo automaticamente

   __  ____ ____ _  _
 / _\/ ___) ___) )( \
/    \___ \___ ) \/ (
\_/\_(____(____|____/
"""

import sys
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent))

# Importar componentes
from core.video_chunker import VideoChunker
from core.activity_filter import ActivityFilter
from core.event_detector import EventDetector
from core.auto_labeler import AutoLabeler
from utils.convert_dav_videos import DAVConverter
from utils.state_manager import StateManager
from utils.helpers import (
    find_files,
    ensure_dir,
    get_video_base_name,
    create_output_structure
)


class AutomatedPipeline:
    """
    Pipeline completamente automatizado
    Processa todos os videos .dav de videos_full/ automaticamente
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa pipeline automatizado
        
        Args:
            config_path: Caminho do arquivo de configuracao
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Diretorios
        self.videos_full_dir = Path(self.config['directories']['videos_full'])
        self.videos_converted_dir = Path(self.config['directories']['videos_converted'])
        self.data_dir = Path(self.config['directories']['data_processing'])
        
        # Garantir que diretorios existem
        ensure_dir(self.videos_full_dir)
        ensure_dir(self.videos_converted_dir)
        ensure_dir(self.data_dir)
        
        # State manager
        state_file = self.config.get('state', {}).get('file', 'pipeline_state.json')
        self.state_manager = StateManager(state_file)
        
        # Componentes
        self.converter = DAVConverter(config_path)
        
        self.logger.info("Pipeline automatizado inicializado")
        self.logger.info(f"Videos fonte: {self.videos_full_dir}")
        self.logger.info(f"Videos convertidos: {self.videos_converted_dir}")
        self.logger.info(f"Dados de processamento: {self.data_dir}")
    
    def run(self):
        """
        Executa pipeline completo automaticamente
        Processa todos os videos .dav encontrados
        """
        self.logger.info("=" * 80)
        self.logger.info("INICIANDO PIPELINE AUTOMATIZADO")
        self.logger.info("=" * 80)
        
        # Mostrar resumo do estado atual
        self.state_manager.print_summary()
        
        # Encontrar videos .dav
        dav_files = find_files(self.videos_full_dir, ['.dav', '.DAV'], recursive=True)
        
        # Se nao houver .dav, verificar se ja existem MP4 convertidos
        if not dav_files:
            self.logger.info(f"Nenhum arquivo .dav encontrado em {self.videos_full_dir}")
            self.logger.info("Verificando arquivos MP4 ja convertidos...")
            
            mp4_files = find_files(str(self.videos_converted_dir), ['.mp4', '.MP4'], recursive=True)
            
            if not mp4_files:
                self.logger.warning("Nenhum arquivo .dav ou .mp4 encontrado")
                self.logger.info("Coloque arquivos .dav na pasta videos_full/ ou MP4 em videos_converted/")
                return
            
            self.logger.info(f"Encontrados {len(mp4_files)} arquivos MP4 convertidos")
            
            # Processar arquivos MP4 diretamente
            for idx, mp4_file in enumerate(mp4_files, 1):
                video_name = self._get_original_dav_name(mp4_file.name)
                
                self.logger.info("\n" + "=" * 80)
                self.logger.info(f"PROCESSANDO MP4 [{idx}/{len(mp4_files)}]: {video_name}")
                self.logger.info("=" * 80)
                
                # Verificar se ja foi completado
                if self.state_manager.is_video_completed(video_name):
                    self.logger.info(f"Video ja processado completamente: {video_name}")
                    continue
                
                # Processar a partir do chunking (conversao ja feita)
                self._process_from_mp4(mp4_file, video_name)
            
            return
        
        self.logger.info(f"Encontrados {len(dav_files)} arquivos .dav")
        
        # Processar cada video
        for idx, dav_file in enumerate(dav_files, 1):
            video_name = dav_file.name
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"PROCESSANDO VIDEO [{idx}/{len(dav_files)}]: {video_name}")
            self.logger.info("=" * 80)
            
            # Verificar se ja foi completado
            if self.state_manager.is_video_completed(video_name):
                self.logger.info(f"Video ja processado completamente: {video_name}")
                continue
            
            # Verificar se falhou anteriormente
            if self.state_manager.is_video_failed(video_name):
                self.logger.warning(f"Video falhou anteriormente: {video_name}")
                
                retry = input(f"Deseja reprocessar {video_name}? (s/n): ").lower().strip()
                if retry == 's':
                    self.state_manager.reset_video(video_name)
                else:
                    continue
            
            # Processar video completo
            try:
                self._process_single_video(dav_file)
            except Exception as e:
                self.logger.error(f"Erro ao processar {video_name}: {e}", exc_info=True)
                self.state_manager.mark_stage_failed(
                    video_name,
                    "pipeline",
                    str(e)
                )
        
        # Resumo final
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE CONCLUIDO")
        self.logger.info("=" * 80)
        self.state_manager.print_summary()
    
    def _process_single_video(self, dav_file: Path):
        """
        Processa um unico video atraves de todos os estagios
        
        Args:
            dav_file: Path do arquivo .dav
        """
        video_name = dav_file.name
        video_base = get_video_base_name(str(dav_file))
        
        # Determinar proximo estagio pendente
        next_stage = self.state_manager.get_next_pending_stage(video_name)
        
        if next_stage is None:
            self.logger.info(f"Todos os estagios concluidos para: {video_name}")
            return
        
        self.logger.info(f"Iniciando do estagio: {next_stage}")
        
        # ESTAGIO 1: CONVERSAO
        if next_stage == StateManager.STAGE_CONVERSION:
            mp4_path = self._run_conversion(dav_file, video_name)
            if mp4_path is None:
                return  # Falhou
            next_stage = StateManager.STAGE_CHUNKING
        else:
            # Encontrar MP4 ja convertido
            mp4_path = self.videos_converted_dir / f"{video_base}_converted.mp4"
            if not mp4_path.exists():
                self.logger.error(f"MP4 nao encontrado: {mp4_path}")
                self.state_manager.mark_stage_failed(
                    video_name,
                    StateManager.STAGE_CONVERSION,
                    "MP4 convertido nao encontrado"
                )
                return
        
        # Criar estrutura de output para este video
        video_data_dir = self.data_dir / video_base
        output_dirs = create_output_structure(str(video_data_dir))
        
        # ESTAGIO 2: CHUNKING
        if next_stage == StateManager.STAGE_CHUNKING:
            chunks = self._run_chunking(mp4_path, output_dirs['chunks'], video_name)
            if chunks is None:
                return  # Falhou
            next_stage = StateManager.STAGE_FILTERING
        else:
            # Carregar chunks ja processados
            chunks_index = output_dirs['chunks'] / 'chunks_index.json'
            if not chunks_index.exists():
                self.logger.error(f"Index de chunks nao encontrado: {chunks_index}")
                self.state_manager.mark_stage_failed(
                    video_name,
                    StateManager.STAGE_CHUNKING,
                    "Index de chunks nao encontrado"
                )
                return
            
            chunker = VideoChunker(use_gpu=self.config.get('chunking', {}).get('use_gpu', False))
            chunks_data = chunker.load_chunks_index(str(chunks_index))
            chunks = chunks_data['chunks']
        
        # ESTAGIO 3: FILTERING
        if next_stage == StateManager.STAGE_FILTERING:
            active_chunks = self._run_filtering(chunks, output_dirs['active_chunks'], video_name)
            if active_chunks is None:
                return  # Falhou
            next_stage = StateManager.STAGE_DETECTION
        else:
            # Carregar active chunks
            active_report = output_dirs['active_chunks'] / 'active_chunks_report.json'
            if not active_report.exists():
                self.logger.error(f"Relatorio de chunks ativos nao encontrado: {active_report}")
                self.state_manager.mark_stage_failed(
                    video_name,
                    StateManager.STAGE_FILTERING,
                    "Relatorio de chunks ativos nao encontrado"
                )
                return
            
            import json
            with open(active_report, 'r') as f:
                report_data = json.load(f)
            active_chunks = report_data['active_chunks']
        
        # ESTAGIO 4: DETECTION
        if next_stage == StateManager.STAGE_DETECTION:
            events = self._run_detection(active_chunks, output_dirs['events'], video_name)
            if events is None:
                return  # Falhou
            next_stage = StateManager.STAGE_LABELING
        else:
            # Carregar events
            events_summary = output_dirs['events'] / 'events_summary.json'
            if not events_summary.exists():
                self.logger.error(f"Resumo de eventos nao encontrado: {events_summary}")
                self.state_manager.mark_stage_failed(
                    video_name,
                    StateManager.STAGE_DETECTION,
                    "Resumo de eventos nao encontrado"
                )
                return
            
            import json
            with open(events_summary, 'r') as f:
                events_data = json.load(f)
            events = events_data['events']
        
        # ESTAGIO 5: LABELING
        if next_stage == StateManager.STAGE_LABELING:
            proposals = self._run_labeling(events, output_dirs['proposals'], video_name)
            if proposals is None:
                return  # Falhou
            next_stage = StateManager.STAGE_REVIEW
        
        # ESTAGIO 6: HUMAN REVIEW (Opcional)
        if next_stage == StateManager.STAGE_REVIEW:
            self.logger.info("\n--- ESTAGIO 6: REVISAO HUMANA ---")
            self.logger.info("Propostas de anotacao geradas!")
            self.logger.info(f"Total de propostas: {len(proposals)}")
            self.logger.info(f"\nPara revisar, execute:")
            self.logger.info(f"  python automated_pipeline/review_gui.py \\")
            self.logger.info(f"    --proposals {output_dirs['proposals']}/proposals_metadata.json \\")
            self.logger.info(f"    --chunks {output_dirs['chunks'].parent / 'active_chunks'}")
            
            # Perguntar se quer abrir GUI agora
            if self._prompt_review_gui():
                review_success = self._run_review_gui(output_dirs['proposals'], output_dirs['chunks'].parent / 'active_chunks', video_name)
                if not review_success:
                    self.logger.warning("Revisao humana pulada ou com erro")
            else:
                self.logger.info("Revisao humana pulada. Execute manualmente quando necessario.")
            
            # Marcar como concluido mesmo se revisao foi pulada
            self.state_manager.mark_stage_complete(
                video_name,
                StateManager.STAGE_REVIEW,
                output_path=str(output_dirs['proposals'] / 'proposals_metadata.json'),
                metadata={'review_skipped': True}
            )
        
        self.logger.info(f"\nVIDEO PROCESSADO COM SUCESSO: {video_name}")
    
    def _run_conversion(self, dav_file: Path, video_name: str) -> Optional[Path]:
        """Executa estagio de conversao"""
        self.logger.info("\n--- ESTAGIO 1: CONVERSAO DAV -> MP4 ---")
        self.state_manager.mark_stage_start(video_name, StateManager.STAGE_CONVERSION)
        
        try:
            success, result = self.converter.convert_video(dav_file)
            
            if success:
                self.state_manager.mark_stage_complete(
                    video_name,
                    StateManager.STAGE_CONVERSION,
                    output_path=result
                )
                return Path(result)
            else:
                self.state_manager.mark_stage_failed(
                    video_name,
                    StateManager.STAGE_CONVERSION,
                    result
                )
                return None
        
        except Exception as e:
            self.logger.error(f"Erro na conversao: {e}", exc_info=True)
            self.state_manager.mark_stage_failed(
                video_name,
                StateManager.STAGE_CONVERSION,
                str(e)
            )
            return None
    
    def _run_chunking(self, mp4_path: Path, output_dir: Path, video_name: str) -> Optional[list]:
        """Executa estagio de chunking"""
        self.logger.info("\n--- ESTAGIO 2: CHUNKING ---")
        self.state_manager.mark_stage_start(video_name, StateManager.STAGE_CHUNKING)
        
        try:
            chunker = VideoChunker(
                chunk_duration_seconds=self.config['chunking']['chunk_duration_seconds'],
                use_gpu=self.config.get('chunking', {}).get('use_gpu', False)
            )
            
            # Manter output_dir como Path para operacoes
            chunks = chunker.chunk_video(
                video_path=str(mp4_path),
                output_dir=str(output_dir),
                start_time=None  # Usar timestamp atual
            )
            
            # Construir caminho do index usando Path
            chunks_index_path = output_dir / 'chunks_index.json'
            
            self.state_manager.mark_stage_complete(
                video_name,
                StateManager.STAGE_CHUNKING,
                output_path=str(chunks_index_path),
                metadata={'total_chunks': len(chunks)}
            )
            return chunks
        
        except Exception as e:
            self.logger.error(f"Erro no chunking: {e}", exc_info=True)
            self.state_manager.mark_stage_failed(
                video_name,
                StateManager.STAGE_CHUNKING,
                str(e)
            )
            return None
    
    def _run_filtering(self, chunks: list, output_dir: Path, video_name: str) -> Optional[list]:
        """Executa estagio de filtering"""
        self.logger.info("\n--- ESTAGIO 3: ACTIVITY FILTERING ---")
        self.state_manager.mark_stage_start(video_name, StateManager.STAGE_FILTERING)
        
        try:
            cfg = self.config['activity_filter']
            
            activity_filter = ActivityFilter(
                motion_threshold=cfg['motion_threshold'],
                min_person_frames=cfg['min_person_frames'],
                person_detection_model=cfg['person_detection_model'],
                motion_sample_rate=cfg['sample_rate_motion'],
                person_sample_rate=cfg['sample_rate_person']
            )
            
            active_chunks, stats = activity_filter.filter_inactive_chunks(
                chunks,
                output_dir=str(output_dir)
            )
            
            # Construir caminho do relatorio usando Path
            report_path = output_dir / 'active_chunks_report.json'
            
            # Calcular porcentagem de reducao
            reduction_percent = 100 - (len(active_chunks) / len(chunks) * 100) if len(chunks) > 0 else 0
            
            self.state_manager.mark_stage_complete(
                video_name,
                StateManager.STAGE_FILTERING,
                output_path=str(report_path),
                metadata={
                    'total_chunks': len(chunks),
                    'active_chunks': len(active_chunks),
                    'reduction_percent': round(reduction_percent, 1)
                }
            )
            return active_chunks
        
        except Exception as e:
            self.logger.error(f"Erro no filtering: {e}", exc_info=True)
            self.state_manager.mark_stage_failed(
                video_name,
                StateManager.STAGE_FILTERING,
                str(e)
            )
            return None
    
    def _run_detection(self, active_chunks: list, output_dir: Path, video_name: str) -> Optional[list]:
        """Executa estagio de detection"""
        self.logger.info("\n--- ESTAGIO 4: EVENT DETECTION ---")
        self.state_manager.mark_stage_start(video_name, StateManager.STAGE_DETECTION)
        
        try:
            cfg = self.config['event_detector']
            
            self.logger.info(f"Carregando EventDetector com modelo {cfg['detector_model']}...")
            
            detector = EventDetector(
                detector_model=cfg['detector_model'],
                tracker_config=cfg['tracker'],
                confidence_threshold=cfg['conf_threshold'],
                min_duration_seconds=cfg['min_event_duration_seconds'],
                sample_rate=cfg.get('sample_rate', 1)  # Pegar do config ou usar 1 (todos os frames)
            )
            
            self.logger.info("EventDetector carregado com sucesso!")
            
            events, stats = detector.detect_events_batch(
                active_chunks,
                output_dir=str(output_dir)
            )
            
            # Construir caminho do relatorio usando Path
            events_path = output_dir / 'events_summary.json'
            
            self.state_manager.mark_stage_complete(
                video_name,
                StateManager.STAGE_DETECTION,
                output_path=str(events_path),
                metadata={'total_events': len(events)}
            )
            return events
        
        except Exception as e:
            self.logger.error(f"Erro na detection: {e}", exc_info=True)
            self.state_manager.mark_stage_failed(
                video_name,
                StateManager.STAGE_DETECTION,
                str(e)
            )
            return None
    
    def _run_labeling(self, events: list, output_dir: Path, video_name: str) -> Optional[list]:
        """Executa estagio de labeling"""
        self.logger.info("\n--- ESTAGIO 5: AUTO LABELING ---")
        self.state_manager.mark_stage_start(video_name, StateManager.STAGE_LABELING)
        
        try:
            cfg = self.config['auto_labeler']['heuristics']
            
            labeler = AutoLabeler(
                normal_max_duration=cfg['normal_duration_max'],
                suspicious_min_duration=cfg['suspicious_duration_min'],
                suspicious_min_frames=cfg['suspicious_frame_threshold']
            )
            
            proposals, stats = labeler.generate_proposals_batch(
                events,
                output_dir=str(output_dir)
            )
            
            # Construir caminho do metadata usando Path
            proposals_path = output_dir / 'proposals_metadata.json'
            
            self.state_manager.mark_stage_complete(
                video_name,
                StateManager.STAGE_LABELING,
                output_path=str(proposals_path),
                metadata={
                    'total_proposals': len(proposals),
                    'needs_review': sum(1 for p in proposals if p['needs_review'])
                }
            )
            return proposals
        
        except Exception as e:
            self.logger.error(f"Erro no labeling: {e}", exc_info=True)
            self.state_manager.mark_stage_failed(
                video_name,
                StateManager.STAGE_LABELING,
                str(e)
            )
            return None
    
    def _prompt_review_gui(self) -> bool:
        """
        Pergunta ao usuario se quer abrir a GUI de revisao agora
        
        Returns:
            True se usuario quer abrir GUI, False caso contrario
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("REVISAO HUMANA DISPONIVEL")
        self.logger.info("=" * 60)
        
        try:
            response = input("\nDeseja abrir a interface de revisao agora? (s/n): ").lower().strip()
            return response in ['s', 'sim', 'y', 'yes']
        except (EOFError, KeyboardInterrupt):
            return False
    
    def _run_review_gui(self, proposals_dir: Path, chunks_dir: Path, video_name: str) -> bool:
        """
        Executa GUI de revisao humana
        
        Args:
            proposals_dir: Diretorio com proposals_metadata.json
            chunks_dir: Diretorio com chunks de video
            video_name: Nome do video sendo processado
            
        Returns:
            True se revisao foi concluida com sucesso, False caso contrario
        """
        self.state_manager.mark_stage_start(video_name, StateManager.STAGE_REVIEW)
        
        try:
            # Importar GUI
            from review_gui import ProposalReviewGUI
            
            proposals_path = proposals_dir / 'proposals_metadata.json'
            
            if not proposals_path.exists():
                self.logger.error(f"Arquivo de propostas nao encontrado: {proposals_path}")
                return False
            
            self.logger.info("Abrindo interface de revisao...")
            
            # Criar e executar GUI
            gui = ProposalReviewGUI(
                proposals_path=str(proposals_path),
                chunks_dir=str(chunks_dir)
            )
            gui.run()
            
            # Verificar se review_results.json foi criado
            annotations_dir = Path('data_processing/annotations')
            results_path = annotations_dir / 'review_results.json'
            
            if results_path.exists():
                self.logger.info("Revisao concluida com sucesso!")
                
                # Carregar resultados
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                summary = results.get('summary', {})
                
                self.state_manager.mark_stage_complete(
                    video_name,
                    StateManager.STAGE_REVIEW,
                    output_path=str(results_path),
                    metadata={
                        'total_reviewed': summary.get('total', 0),
                        'approved': summary.get('approved', 0),
                        'rejected': summary.get('rejected', 0),
                        'corrected': summary.get('corrected', 0),
                        'approval_rate': summary.get('approval_rate', 0)
                    }
                )
                return True
            else:
                self.logger.warning("GUI fechada sem salvar resultados")
                return False
        
        except ImportError as e:
            self.logger.error(f"Erro ao importar review_gui: {e}")
            self.logger.info("Execute: pip install pillow")
            return False
        except Exception as e:
            self.logger.error(f"Erro ao executar GUI de revisao: {e}", exc_info=True)
            self.state_manager.mark_stage_failed(
                video_name,
                StateManager.STAGE_REVIEW,
                str(e)
            )
            return False
    
    def _load_config(self, config_path: str) -> dict:
        """Carrega configuracao do arquivo YAML"""
        config_file = Path(__file__).parent / config_path
        
        if not config_file.exists():
            self.logger.error(f"Arquivo de configuracao nao encontrado: {config_file}")
            sys.exit(1)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Erro ao carregar configuracao: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> logging.Logger:
        """Configura sistema de logging"""
        logger = logging.getLogger('automated_pipeline')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('automated_pipeline.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def _get_original_dav_name(self, mp4_filename: str) -> str:
        """
        Extrai nome original do .dav a partir do nome do MP4 convertido
        
        Args:
            mp4_filename: Nome do arquivo MP4 (ex: video_converted.mp4)
            
        Returns:
            Nome original do .dav (ex: video.dav)
        """
        # Remove sufixo _converted.mp4 e adiciona .dav
        if mp4_filename.endswith('_converted.mp4'):
            base_name = mp4_filename[:-len('_converted.mp4')]
            return f"{base_name}.dav"
        else:
            # Fallback: substitui .mp4 por .dav
            return mp4_filename.replace('.mp4', '.dav')
    
    def _process_from_mp4(self, mp4_file: Path, video_name: str):
        """
        Processa video a partir do MP4 ja convertido
        Comeca do estagio de chunking
        
        Args:
            mp4_file: Path do arquivo MP4 convertido
            video_name: Nome original do video (.dav)
        """
        video_base = get_video_base_name(video_name)
        
        # Determinar proximo estagio pendente (deve ser chunking ou posterior)
        next_stage = self.state_manager.get_next_pending_stage(video_name)
        
        if next_stage is None:
            self.logger.info(f"Todos os estagios concluidos para: {video_name}")
            return
        
        # Se conversao ainda nao foi marcada como concluida, marca agora
        if next_stage == StateManager.STAGE_CONVERSION:
            self.state_manager.mark_stage_complete(
                video_name,
                StateManager.STAGE_CONVERSION,
                str(mp4_file),
                {"conversion_skipped": True}
            )
            next_stage = StateManager.STAGE_CHUNKING
        
        self.logger.info(f"Iniciando do estagio: {next_stage}")
        
        # Criar diretorios de output
        output_dirs = create_output_structure(str(self.data_dir))
        
        # ESTAGIO 2: CHUNKING
        if next_stage == StateManager.STAGE_CHUNKING:
            chunks = self._run_chunking(mp4_file, output_dirs['chunks'], video_name)
            if chunks is None:
                return  # Falhou
            next_stage = StateManager.STAGE_FILTERING
        
        # ESTAGIO 3: FILTRAGEM
        if next_stage == StateManager.STAGE_FILTERING:
            # Carregar lista de metadados dos chunks
            chunks_index_path = output_dirs['chunks'] / 'chunks_index.json'
            if not chunks_index_path.exists():
                self.logger.error(f"Arquivo de index de chunks nao encontrado: {chunks_index_path}")
                return
            with open(chunks_index_path, 'r', encoding='utf-8') as f:
                chunks_index = json.load(f)
            chunks_metadata = chunks_index['chunks']
            active_chunks = self._run_filtering(chunks_metadata, output_dirs['active_chunks'], video_name)
            if active_chunks is None:
                return  # Falhou
            next_stage = StateManager.STAGE_DETECTION
        
        # ESTAGIO 4: DETECCAO
        if next_stage == StateManager.STAGE_DETECTION:
            # Carregar chunks ativos do relatorio JSON
            active_chunks_report_path = output_dirs['active_chunks'] / 'active_chunks_report.json'
            with open(active_chunks_report_path, 'r', encoding='utf-8') as f:
                active_report = json.load(f)
            active_chunks_list = active_report['active_chunks']
            
            events = self._run_detection(active_chunks_list, output_dirs['events'], video_name)
            if events is None:
                return  # Falhou
            next_stage = StateManager.STAGE_LABELING
        
        # ESTAGIO 5: ROTULAGEM
        if next_stage == StateManager.STAGE_LABELING:
            # Carregar eventos do relatorio JSON
            events_summary_path = output_dirs['events'] / 'events_summary.json'
            with open(events_summary_path, 'r', encoding='utf-8') as f:
                events_data = json.load(f)
            events_list = events_data['events']
            
            proposals = self._run_labeling(events_list, output_dirs['proposals'], video_name)
            if proposals is None:
                return  # Falhou
        
        self.logger.info(f"✅ Processamento completo para: {video_name}")


def main():
    """Funcao principal"""
    print(r"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  PIPELINE AUTOMATIZADO DE DETECCAO DE FURTOS                 ║
    ║  Processamento automatico de videos de vigilancia            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        pipeline = AutomatedPipeline()
        pipeline.run()
        
        print("\n✅ Pipeline concluido com sucesso!")
        print(f"📊 Verifique o estado em: pipeline_state.json")
        print(f"📁 Dados processados em: data_processing/")
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrompido pelo usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        logging.exception("Erro fatal no pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
