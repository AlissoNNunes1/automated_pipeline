r"""
StateManager: Gerencia estado do processamento de videos

Rastreia progresso de cada video .dav atraves do pipeline completo
para evitar reprocessamento e permitir retomada de onde parou

Estrutura do estado:
{
    "video_name.dav": {
        "status": "processing" | "completed" | "failed",
        "stages": {
            "conversion": {"status": "completed", "timestamp": "...", "output": "..."},
            "chunking": {"status": "completed", "timestamp": "...", "output": "..."},
            "filtering": {"status": "processing", "timestamp": "..."},
            "detection": {"status": "not_started"},
            "labeling": {"status": "not_started"}
        },
        "error": null | "mensagem de erro",
        "started_at": "2025-10-09 10:00:00",
        "completed_at": null | "2025-10-09 15:30:00"
    }
}

   __  ____ ____ _  _
 / _\/ ___) ___) )( \
/    \___ \___ ) \/ (
\_/\_(____(____|____/
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from threading import Lock


class StateManager:
    """
    Gerencia estado do processamento de videos
    Permite rastreamento de progresso e retomada de processamento
    """

    # Estados possiveis
    STATUS_NOT_STARTED = "not_started"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    # Estagios do pipeline
    STAGE_CONVERSION = "conversion"
    STAGE_CHUNKING = "chunking"
    STAGE_FILTERING = "filtering"
    STAGE_DETECTION = "detection"
    STAGE_LABELING = "labeling"
    STAGE_REVIEW = "review"

    ALL_STAGES = [
        STAGE_CONVERSION,
        STAGE_CHUNKING,
        STAGE_FILTERING,
        STAGE_DETECTION,
        STAGE_LABELING,
        STAGE_REVIEW
    ]

    def __init__(self, state_file: str = "pipeline_state.json"):
        """
        Inicializa StateManager

        Args:
            state_file: Caminho do arquivo JSON de estado
        """
        self.state_file = Path(state_file)
        self.logger = logging.getLogger(__name__)
        self._lock = Lock()  # Thread-safe
        
        # Carregar estado existente ou criar novo
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Carrega estado do arquivo JSON"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.logger.info(f"Estado carregado de: {self.state_file}")
                return state
            except Exception as e:
                self.logger.error(f"Erro ao carregar estado: {e}")
                return {}
        else:
            self.logger.info("Nenhum estado anterior encontrado, criando novo")
            return {}

    def _save_state(self):
        """Salva estado no arquivo JSON"""
        try:
            with self._lock:
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Estado salvo em: {self.state_file}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {e}")

    def initialize_video(self, video_name: str):
        """
        Inicializa estado de um novo video

        Args:
            video_name: Nome do arquivo .dav
        """
        if video_name not in self.state:
            self.state[video_name] = {
                'status': self.STATUS_NOT_STARTED,
                'stages': {
                    stage: {'status': self.STATUS_NOT_STARTED}
                    for stage in self.ALL_STAGES
                },
                'error': None,
                'started_at': datetime.now().isoformat(),
                'completed_at': None
            }
            self._save_state()
            self.logger.info(f"Video inicializado: {video_name}")

    def mark_stage_start(self, video_name: str, stage: str):
        """
        Marca inicio de um estagio

        Args:
            video_name: Nome do video
            stage: Nome do estagio
        """
        if video_name not in self.state:
            self.initialize_video(video_name)

        self.state[video_name]['stages'][stage] = {
            'status': self.STATUS_PROCESSING,
            'started_at': datetime.now().isoformat()
        }
        self.state[video_name]['status'] = self.STATUS_PROCESSING
        self._save_state()
        self.logger.info(f"[{video_name}] Iniciando estagio: {stage}")

    def mark_stage_complete(
        self,
        video_name: str,
        stage: str,
        output_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Marca conclusao de um estagio

        Args:
            video_name: Nome do video
            stage: Nome do estagio
            output_path: Caminho do output gerado (opcional)
            metadata: Metadata adicional do estagio (opcional)
        """
        if video_name not in self.state:
            self.initialize_video(video_name)

        stage_data = self.state[video_name]['stages'][stage]
        stage_data['status'] = self.STATUS_COMPLETED
        stage_data['completed_at'] = datetime.now().isoformat()
        
        if output_path:
            stage_data['output'] = output_path
        
        if metadata:
            stage_data['metadata'] = metadata

        # Verificar se todos os estagios foram concluidos
        all_completed = all(
            self.state[video_name]['stages'][s]['status'] == self.STATUS_COMPLETED
            for s in self.ALL_STAGES
        )

        if all_completed:
            self.state[video_name]['status'] = self.STATUS_COMPLETED
            self.state[video_name]['completed_at'] = datetime.now().isoformat()
            self.logger.info(f"[{video_name}] PIPELINE COMPLETO!")
        
        self._save_state()
        self.logger.info(f"[{video_name}] Estagio concluido: {stage}")

    def mark_stage_failed(
        self,
        video_name: str,
        stage: str,
        error_message: str
    ):
        """
        Marca falha de um estagio

        Args:
            video_name: Nome do video
            stage: Nome do estagio
            error_message: Mensagem de erro
        """
        if video_name not in self.state:
            self.initialize_video(video_name)

        self.state[video_name]['stages'][stage] = {
            'status': self.STATUS_FAILED,
            'error': error_message,
            'failed_at': datetime.now().isoformat()
        }
        self.state[video_name]['status'] = self.STATUS_FAILED
        self.state[video_name]['error'] = f"[{stage}] {error_message}"
        
        self._save_state()
        self.logger.error(f"[{video_name}] Falha no estagio {stage}: {error_message}")

    def get_video_status(self, video_name: str) -> Optional[Dict]:
        """
        Obtem status de um video

        Args:
            video_name: Nome do video

        Returns:
            Dict com status do video ou None se nao encontrado
        """
        return self.state.get(video_name)

    def get_next_pending_stage(self, video_name: str) -> Optional[str]:
        """
        Obtem proximo estagio pendente para um video

        Args:
            video_name: Nome do video

        Returns:
            Nome do proximo estagio ou None se todos completos/falhou
        """
        if video_name not in self.state:
            return self.STAGE_CONVERSION  # Comecar do inicio

        video_state = self.state[video_name]

        # Se completou ou falhou, nao ha pendencias
        if video_state['status'] in [self.STATUS_COMPLETED, self.STATUS_FAILED]:
            return None

        # Encontrar primeiro estagio nao completo
        for stage in self.ALL_STAGES:
            stage_status = video_state['stages'][stage]['status']
            if stage_status != self.STATUS_COMPLETED:
                return stage

        return None

    def is_video_completed(self, video_name: str) -> bool:
        """
        Verifica se video completou pipeline

        Args:
            video_name: Nome do video

        Returns:
            True se completou, False caso contrario
        """
        if video_name not in self.state:
            return False

        return self.state[video_name]['status'] == self.STATUS_COMPLETED

    def is_video_failed(self, video_name: str) -> bool:
        """
        Verifica se video falhou no pipeline

        Args:
            video_name: Nome do video

        Returns:
            True se falhou, False caso contrario
        """
        if video_name not in self.state:
            return False

        return self.state[video_name]['status'] == self.STATUS_FAILED

    def get_all_videos(self) -> List[str]:
        """
        Obtem lista de todos os videos no estado

        Returns:
            Lista de nomes de videos
        """
        return list(self.state.keys())

    def get_videos_by_status(self, status: str) -> List[str]:
        """
        Obtem videos com determinado status

        Args:
            status: Status desejado

        Returns:
            Lista de nomes de videos
        """
        return [
            video_name
            for video_name, video_state in self.state.items()
            if video_state['status'] == status
        ]

    def get_statistics(self) -> Dict:
        """
        Obtem estatisticas gerais do processamento

        Returns:
            Dict com estatisticas
        """
        total = len(self.state)
        completed = len(self.get_videos_by_status(self.STATUS_COMPLETED))
        processing = len(self.get_videos_by_status(self.STATUS_PROCESSING))
        failed = len(self.get_videos_by_status(self.STATUS_FAILED))
        not_started = len(self.get_videos_by_status(self.STATUS_NOT_STARTED))

        return {
            'total_videos': total,
            'completed': completed,
            'processing': processing,
            'failed': failed,
            'not_started': not_started,
            'completion_rate': f"{(completed/total*100):.1f}%" if total > 0 else "0%"
        }

    def reset_video(self, video_name: str):
        """
        Reseta estado de um video (para reprocessamento)

        Args:
            video_name: Nome do video
        """
        if video_name in self.state:
            del self.state[video_name]
            self._save_state()
            self.logger.info(f"Video resetado: {video_name}")

    def reset_failed_videos(self):
        """Reseta todos os videos que falharam"""
        failed_videos = self.get_videos_by_status(self.STATUS_FAILED)
        for video_name in failed_videos:
            self.reset_video(video_name)
        self.logger.info(f"Resetados {len(failed_videos)} videos com falha")

    def reset_stages_from(self, video_name: str, from_stage: str):
        """Reseta estagios a partir de um estagio informado (inclusive)
        
        Comentario sem acento: usado para modo start-from, para garantir que o
        estagio escolhido e os posteriores sejam marcados como not_started e
        possam ter seus status atualizados durante a nova execucao.
        """
        if video_name not in self.state:
            self.initialize_video(video_name)

        # Encontrar indice do estagio
        try:
            start_idx = self.ALL_STAGES.index(from_stage)
        except ValueError:
            self.logger.warning(f"Estagio invalido em reset_stages_from: {from_stage}")
            return

        # Resetar estagios do indice em diante
        for stage in self.ALL_STAGES[start_idx:]:
            self.state[video_name]['stages'][stage] = {
                'status': self.STATUS_NOT_STARTED
            }

        # Ajustar status de nivel superior para permitir novo processamento
        self.state[video_name]['status'] = self.STATUS_NOT_STARTED
        self.state[video_name]['error'] = None
        self.state[video_name]['completed_at'] = None
        self._save_state()
        self.logger.info(f"[{video_name}] Estagios resetados a partir de: {from_stage}")

    def reset_stage_only(self, video_name: str, stage: str):
        """Reseta somente um estagio, preservando os demais
        
        Comentario sem acento: usado para modo run-stage, para que o estagio
        seja executado novamente sem afetar os outros.
        """
        if video_name not in self.state:
            self.initialize_video(video_name)

        if stage not in self.ALL_STAGES:
            self.logger.warning(f"Estagio invalido em reset_stage_only: {stage}")
            return

        self.state[video_name]['stages'][stage] = {
            'status': self.STATUS_NOT_STARTED
        }
        # Nao altera status de nivel superior aqui
        self._save_state()
        self.logger.info(f"[{video_name}] Estagio resetado: {stage}")

    def print_summary(self):
        """Imprime resumo do estado atual"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("RESUMO DO ESTADO DO PIPELINE")
        print("=" * 60)
        print(f"Total de videos: {stats['total_videos']}")
        print(f"Completados: {stats['completed']} ({stats['completion_rate']})")
        print(f"Em processamento: {stats['processing']}")
        print(f"Falharam: {stats['failed']}")
        print(f"Nao iniciados: {stats['not_started']}")
        print("=" * 60)

        # Mostrar videos em processamento
        processing = self.get_videos_by_status(self.STATUS_PROCESSING)
        if processing:
            print("\nVideos em processamento:")
            for video_name in processing:
                video_state = self.state[video_name]
                current_stage = None
                for stage in self.ALL_STAGES:
                    if video_state['stages'][stage]['status'] == self.STATUS_PROCESSING:
                        current_stage = stage
                        break
                print(f"  - {video_name} [{current_stage}]")

        # Mostrar videos que falharam
        failed = self.get_videos_by_status(self.STATUS_FAILED)
        if failed:
            print("\nVideos que falharam:")
            for video_name in failed:
                video_state = self.state[video_name]
                # Tentar mostrar erro de nivel superior; se ausente, buscar no estagio que falhou
                err_msg = video_state.get('error')
                if not err_msg:
                    # Percorrer estagios na ordem e capturar o primeiro com status failed
                    for stage in self.ALL_STAGES:
                        st_data = video_state.get('stages', {}).get(stage, {})
                        if st_data.get('status') == self.STATUS_FAILED:
                            # Montar mensagem com nome do estagio e erro registrado, se houver
                            stage_err = st_data.get('error') or 'falha sem mensagem'
                            err_msg = f"[{stage}] {stage_err}"
                            break
                print(f"  - {video_name}: {err_msg or 'Erro desconhecido'}")

        print()


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
