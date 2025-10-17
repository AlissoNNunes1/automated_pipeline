"""
Interface de Revisao Humana - Pipeline Automatizado
Substitui CVAT por GUI local simples e rapida

Funcionalidades:
- Preview de frames com bboxes
- Playback de eventos completos
- Teclas rapidas para aprovacao/rejeicao
- Correcao de classes
- Export automatico para YOLO format
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProposalReviewGUI:
    """
    Interface grafica para revisar propostas de anotacao
    Teclas rapidas: A (aprovar), R (rejeitar), C (corrigir), P (play)
    """
    
    # Classes do modelo
    CLASSES = [
        'comportamento_normal',
        'furto_discreto',
        'furto_bolsa_mochila',
        'furto_grupo_colaborativo',
        'acoes_ambiguas_suspeitas',
        'funcionario_reposicao'
    ]
    
    # Cores para classes (BGR para OpenCV)
    CLASS_COLORS = {
        'comportamento_normal': (0, 255, 0),           # Verde
        'furto_discreto': (0, 0, 255),                 # Vermelho
        'furto_bolsa_mochila': (0, 165, 255),          # Laranja
        'furto_grupo_colaborativo': (255, 0, 255),     # Magenta
        'acoes_ambiguas_suspeitas': (0, 255, 255),     # Amarelo
        'funcionario_reposicao': (255, 255, 0)         # Ciano
    }
    
    def __init__(self, proposals_path: str, chunks_dir: str = "data_processing/active_chunks", progress_file: str = None):
        """
        Inicializa interface de revisao
        
        Args:
            proposals_path: Caminho para proposals_metadata.json
            chunks_dir: Diretorio com chunks de video
            progress_file: Arquivo para salvar progresso (opcional)
        """
        self.proposals_path = proposals_path
        self.chunks_dir = Path(chunks_dir)
        
        # Auto-detectar diretorio de chunks correto
        self.chunks_dir = self._find_chunks_directory(proposals_path, chunks_dir)
        
        # Arquivo de progresso
        if progress_file is None:
            # Usar mesmo diretorio do proposals com nome review_progress.json
            progress_file = Path(proposals_path).parent / 'review_progress.json'
        self.progress_file = Path(progress_file)
        
        # Carregar propostas
        self.proposals = self._load_proposals()
        self.current_idx = 0
        
        # Resultados da revisao
        self.approved = []
        self.rejected = []
        self.corrected = []
        
        # Carregar progresso anterior se existir
        self._load_progress()
        
        # Estado do video player
        self.playing = False
        self.current_video_cap = None
        
        # Setup GUI
        self._setup_gui()
        
        logger.info(f"Carregadas {len(self.proposals)} propostas para revisao")
        logger.info(f"Chunks directory: {self.chunks_dir}")
    
    def _find_chunks_directory(self, proposals_path: str, chunks_dir_hint: str) -> Path:
        """
        Encontra diretorio correto de chunks baseado no proposals_path
        
        Args:
            proposals_path: Caminho do arquivo de propostas
            chunks_dir_hint: Sugestao de diretorio (pode estar errado)
            
        Returns:
            Path do diretorio correto com chunks
        """
        proposals_dir = Path(proposals_path).parent
        
        # Tentar varios caminhos possiveis
        possible_paths = [
            # 1. Hint fornecido
            Path(chunks_dir_hint),
            
            # 2. Subir 1 nivel e procurar chunks/
            proposals_dir.parent / 'chunks',
            
            # 3. Subir 2 niveis e procurar chunks/
            proposals_dir.parent.parent / 'chunks',
            
            # 4. Mesmo nivel de proposals/
            proposals_dir.parent / 'active_chunks',
            
            # 5. Subir e procurar active_chunks/
            proposals_dir.parent.parent / 'active_chunks',
            
            # 6. Caminho completo automated_pipeline/data_processing/1/chunks
            Path('automated_pipeline/data_processing/1/chunks'),
            
            # 7. Relativo ao proposals: ../chunks
            proposals_dir.parent / 'chunks',
        ]
        
        # Testar cada caminho
        for path in possible_paths:
            if path.exists() and path.is_dir():
                # Verificar se tem arquivos .mp4
                mp4_files = list(path.glob('*.mp4'))
                if mp4_files:
                    logger.info(f"Chunks directory encontrado: {path} ({len(mp4_files)} chunks)")
                    return path
        
        # Se nenhum funcionou, usar hint original e avisar
        logger.warning(f"Nenhum diretorio de chunks com .mp4 encontrado. Usando: {chunks_dir_hint}")
        logger.warning(f"Caminhos testados: {[str(p) for p in possible_paths]}")
        return Path(chunks_dir_hint)
    
    def _load_progress(self):
        """Carrega progresso de revisao anterior"""
        if not self.progress_file.exists():
            logger.info("Nenhum progresso anterior encontrado. Iniciando do zero.")
            return
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            
            # Restaurar estado
            self.current_idx = progress.get('current_idx', 0)
            self.approved = progress.get('approved', [])
            self.rejected = progress.get('rejected', [])
            self.corrected = progress.get('corrected', [])
            
            logger.info(f"Progresso carregado: Proposta {self.current_idx + 1}/{len(self.proposals)}")
            logger.info(f"  Aprovadas: {len(self.approved)}, Rejeitadas: {len(self.rejected)}, Corrigidas: {len(self.corrected)}")
            
            # Mostrar mensagem ao usuario
            if self.current_idx > 0:
                import tkinter.messagebox as mb
                mb.showinfo(
                    "Progresso Restaurado",
                    f"Sessao anterior encontrada!\n\n"
                    f"Continuando da proposta {self.current_idx + 1}/{len(self.proposals)}\n"
                    f"Aprovadas: {len(self.approved)}\n"
                    f"Rejeitadas: {len(self.rejected)}\n"
                    f"Corrigidas: {len(self.corrected)}"
                )
        
        except Exception as e:
            logger.error(f"Erro ao carregar progresso: {e}")
            logger.info("Iniciando revisao do zero")
    
    def _save_progress(self):
        """Salva progresso atual da revisao"""
        try:
            progress = {
                'current_idx': self.current_idx,
                'approved': self.approved,
                'rejected': self.rejected,
                'corrected': self.corrected,
                'last_save': datetime.now().isoformat(),
                'proposals_file': str(self.proposals_path),
                'total_proposals': len(self.proposals)
            }
            
            # Criar diretorio se necessario
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Progresso salvo: {self.progress_file}")
        
        except Exception as e:
            logger.error(f"Erro ao salvar progresso: {e}")
    
    def _delete_progress(self):
        """Remove arquivo de progresso apos finalizacao"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                logger.info(f"Arquivo de progresso removido: {self.progress_file}")
        except Exception as e:
            logger.warning(f"Erro ao remover progresso: {e}")
        logger.info(f"Chunks directory: {self.chunks_dir}")
    
    def _load_proposals(self) -> List[Dict]:
        """Carrega propostas do arquivo JSON"""
        try:
            with open(self.proposals_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Verificar formato
            if isinstance(data, dict) and 'proposals' in data:
                proposals = data['proposals']
            elif isinstance(data, list):
                proposals = data
            else:
                raise ValueError("Formato de proposals invalido")
            
            logger.info(f"Carregadas {len(proposals)} propostas de {self.proposals_path}")
            return proposals
            
        except FileNotFoundError:
            logger.error(f"Arquivo nao encontrado: {self.proposals_path}")
            messagebox.showerror("Erro", f"Arquivo nao encontrado:\n{self.proposals_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON: {e}")
            messagebox.showerror("Erro", f"Erro ao ler JSON:\n{str(e)}")
            raise
    
    def _setup_gui(self):
        """Configura interface grafica"""
        self.root = tk.Tk()
        self.root.title("Pipeline Review - Validacao de Propostas")
        self.root.geometry("1600x950")
        self.root.configure(bg='#2b2b2b')
        
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', padding=6, relief="flat", background="#4CAF50")
        
        # Container principal
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # === HEADER ===
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(
            header_frame,
            text="🎥 Pipeline Review - Validacao de Eventos",
            font=("Arial", 16, "bold"),
            bg='#2b2b2b',
            fg='white'
        )
        title_label.pack(side=tk.LEFT)
        
        # Progress no header
        self.header_progress_label = tk.Label(
            header_frame,
            text="Proposta 0/0 (0%)",
            font=("Arial", 12),
            bg='#2b2b2b',
            fg='#4CAF50'
        )
        self.header_progress_label.pack(side=tk.RIGHT)
        
        # === CONTENT AREA (2 colunas) ===
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Coluna esquerda: Video e controles
        left_column = ttk.Frame(content_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        video_frame = ttk.LabelFrame(left_column, text="Preview do Evento", padding="10")
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Controles de navegacao
        nav_frame = ttk.Frame(left_column)
        nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            nav_frame, 
            text="◀ Anterior (←)", 
            command=self.previous_proposal
        ).pack(side=tk.LEFT, padx=5)
        
        self.position_label = tk.Label(
            nav_frame,
            text="Frame 0/0",
            font=("Arial", 10),
            bg='#2b2b2b',
            fg='white'
        )
        self.position_label.pack(side=tk.LEFT, expand=True)
        
        ttk.Button(
            nav_frame, 
            text="Proximo (→) ▶", 
            command=self.next_proposal
        ).pack(side=tk.RIGHT, padx=5)
        
        # Coluna direita: Info e controles
        right_column = ttk.Frame(content_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        
        # Info panel
        info_frame = ttk.LabelFrame(right_column, text="Informacoes do Evento", padding="10")
        info_frame.pack(fill=tk.BOTH, pady=(0, 10))
        
        self.info_text = tk.Text(
            info_frame, 
            height=12, 
            width=50,
            bg='#1e1e1e',
            fg='white',
            font=("Consolas", 10),
            wrap=tk.WORD
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Classes panel
        classes_frame = ttk.LabelFrame(right_column, text="Classificacao", padding="10")
        classes_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.class_var = tk.StringVar()
        
        for i, cls in enumerate(self.CLASSES):
            rb = ttk.Radiobutton(
                classes_frame,
                text=cls.replace('_', ' ').title(),
                variable=self.class_var,
                value=cls
            )
            rb.pack(anchor=tk.W, pady=2)
        
        # Action buttons
        actions_frame = ttk.LabelFrame(right_column, text="Acoes", padding="10")
        actions_frame.pack(fill=tk.X)
        
        # Botao Play
        self.play_button = tk.Button(
            actions_frame,
            text="▶ Play Video (P)",
            command=self.play_video,
            bg='#2196F3',
            fg='white',
            font=("Arial", 11, "bold"),
            relief=tk.RAISED,
            bd=2
        )
        self.play_button.pack(fill=tk.X, pady=5)
        
        # Botao Aprovar
        approve_btn = tk.Button(
            actions_frame,
            text="✓ Aprovar (A)",
            command=self.approve,
            bg='#4CAF50',
            fg='white',
            font=("Arial", 11, "bold"),
            relief=tk.RAISED,
            bd=2
        )
        approve_btn.pack(fill=tk.X, pady=5)
        
        # Botao Corrigir
        correct_btn = tk.Button(
            actions_frame,
            text="✎ Corrigir Classe (C)",
            command=self.correct_class,
            bg='#FF9800',
            fg='white',
            font=("Arial", 11, "bold"),
            relief=tk.RAISED,
            bd=2
        )
        correct_btn.pack(fill=tk.X, pady=5)
        
        # Botao Rejeitar
        reject_btn = tk.Button(
            actions_frame,
            text="✗ Rejeitar (R)",
            command=self.reject,
            bg='#f44336',
            fg='white',
            font=("Arial", 11, "bold"),
            relief=tk.RAISED,
            bd=2
        )
        reject_btn.pack(fill=tk.X, pady=5)
        
        # === FOOTER ===
        footer_frame = ttk.Frame(main_container)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Stats
        self.stats_label = tk.Label(
            footer_frame,
            text="Aprovadas: 0 | Rejeitadas: 0 | Corrigidas: 0",
            font=("Arial", 10),
            bg='#2b2b2b',
            fg='#888'
        )
        self.stats_label.pack(side=tk.LEFT)
        
        # Botao Finalizar
        ttk.Button(
            footer_frame,
            text="💾 Finalizar e Exportar",
            command=self.finish_review
        ).pack(side=tk.RIGHT)
        
        # === KEYBOARD SHORTCUTS ===
        self.root.bind('a', lambda e: self.approve())
        self.root.bind('A', lambda e: self.approve())
        self.root.bind('r', lambda e: self.reject())
        self.root.bind('R', lambda e: self.reject())
        self.root.bind('c', lambda e: self.correct_class())
        self.root.bind('C', lambda e: self.correct_class())
        self.root.bind('p', lambda e: self.play_video())
        self.root.bind('P', lambda e: self.play_video())
        self.root.bind('<Left>', lambda e: self.previous_proposal())
        self.root.bind('<Right>', lambda e: self.next_proposal())
        self.root.bind('<Escape>', lambda e: self.stop_playback())
        
        # Cleanup ao fechar
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Carregar primeira proposta
        if self.proposals:
            self.load_proposal(0)
    
    def load_proposal(self, idx: int):
        """Carrega e exibe proposta"""
        if idx < 0 or idx >= len(self.proposals):
            return
        
        self.current_idx = idx
        proposal = self.proposals[idx]
        
        # Atualizar info text
        self.info_text.delete(1.0, tk.END)
        
        # Obter metadata (pode estar direto no proposal ou em 'metadata')
        metadata = proposal.get('metadata', {})
        event_chars = proposal.get('event_characteristics', {})
        
        # Campos que podem estar em varios lugares
        duration = event_chars.get('duration_seconds') or metadata.get('duration_seconds', 0)
        frame_count = event_chars.get('frame_count') or metadata.get('frame_count', 0)
        track_id = proposal.get('track_id') or metadata.get('track_id', 'N/A')
        chunk_id = proposal.get('chunk_id') or metadata.get('chunk_id', 'N/A')
        confidence_avg = event_chars.get('confidence_avg') or proposal.get('confidence', 0)
        
        info = f"""
╔══════════════════════════════════════════╗
║           DETALHES DO EVENTO             ║
╚══════════════════════════════════════════╝

Event ID: {proposal.get('event_id', 'N/A')}
Chunk ID: {chunk_id}

⏱ Duracao: {duration:.1f}s
🎬 Frames: {frame_count}
🆔 Track ID: {track_id}

📊 Confianca Deteccao: {confidence_avg:.2f}
📈 Confianca Classificacao: {proposal.get('classification_confidence', 0):.2f}

🏷 Classe Sugerida:
   {proposal.get('suggested_class', 'N/A').replace('_', ' ').title()}

💭 Raciocinio:
   {proposal.get('reasoning', 'N/A')}

⚠ Precisa Revisao: {'Sim' if proposal.get('needs_review', False) else 'Nao'}
        """
        
        self.info_text.insert(1.0, info.strip())
        
        # Selecionar classe sugerida
        self.class_var.set(proposal.get('suggested_class', self.CLASSES[0]))
        
        # Carregar frame representativo
        self._load_representative_frame(proposal)
        
        # Atualizar progress
        progress_pct = ((idx + 1) / len(self.proposals)) * 100
        progress_text = f"Proposta {idx+1}/{len(self.proposals)} ({progress_pct:.1f}%)"
        self.header_progress_label.config(text=progress_text)
        self.position_label.config(text=f"Proposta {idx+1} de {len(self.proposals)}")
        
        # Atualizar stats
        self._update_stats()
    
    def _load_representative_frame(self, proposal: Dict):
        """Carrega frame do meio do evento com bbox desenhada"""
        try:
            # Pegar frame do meio
            images = proposal.get('images', [])
            if not images:
                # Se nao tem images, tentar usar bbox_sequence
                bbox_sequence = proposal.get('bbox_sequence', [])
                if not bbox_sequence:
                    self._show_no_image_message()
                    return
                
                # Criar estrutura de images falsa baseada em bbox_sequence
                images = [
                    {'id': i, 'file_name': f'frame_{i:06d}.jpg'}
                    for i in range(len(bbox_sequence))
                ]
            
            mid_idx = len(images) // 2
            mid_frame_info = images[mid_idx]
            
            # Obter caminho do chunk
            chunk_path = self._get_chunk_path(proposal)
            
            # Log de diagnostico
            logger.info(f"Tentando carregar frame de: {chunk_path}")
            logger.info(f"  Chunk ID: {proposal.get('chunk_id', 'N/A')}")
            logger.info(f"  Event ID: {proposal.get('event_id', 'N/A')}")
            logger.info(f"  Chunks dir: {self.chunks_dir}")
            
            if not chunk_path.exists():
                logger.warning(f"Chunk nao encontrado: {chunk_path}")
                
                # Listar chunks disponiveis
                available_chunks = list(self.chunks_dir.glob('*.mp4'))
                logger.info(f"Chunks disponiveis em {self.chunks_dir}:")
                for chunk in available_chunks[:10]:  # Mostrar primeiros 10
                    logger.info(f"  - {chunk.name}")
                
                self._show_no_image_message(f"Video nao encontrado:\n{chunk_path.name}\n\nChunks dir: {self.chunks_dir}")
                return
            
            # Carregar frame
            cap = cv2.VideoCapture(str(chunk_path))
            
            # Determinar frame_id
            if 'id' in mid_frame_info:
                frame_id = mid_frame_info['id']
            else:
                # Se nao tem ID, usar indice do meio
                frame_id = mid_idx
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self._show_no_image_message("Erro ao ler frame")
                return
            
            # Desenhar bbox
            # Tentar primeiro annotations (formato CVAT)
            annotations = proposal.get('annotations', [])
            if annotations:
                annotation = next(
                    (a for a in annotations if a.get('image_id') == frame_id),
                    None
                )
                
                if annotation:
                    bbox = annotation.get('bbox', [])
                    if len(bbox) == 4:
                        self._draw_bbox_on_frame(frame, bbox, proposal)
            
            # Se nao tem annotations, tentar bbox_sequence (formato proposals_metadata.json)
            elif 'bbox_sequence' in proposal:
                bbox_sequence = proposal['bbox_sequence']
                if mid_idx < len(bbox_sequence):
                    bbox = bbox_sequence[mid_idx]
                    if len(bbox) == 4:
                        self._draw_bbox_on_frame(frame, bbox, proposal)
            
            # Exibir frame
            self._display_frame(frame)
            
        except Exception as e:
            logger.error(f"Erro ao carregar frame: {e}")
            self._show_no_image_message(f"Erro: {str(e)}")
    
    def _draw_bbox_on_frame(self, frame, bbox, proposal: Dict):
        """
        Desenha bounding box no frame
        
        Args:
            frame: Frame OpenCV (numpy array)
            bbox: Lista [x1, y1, x2, y2]
            proposal: Dicionario da proposta (para obter cor e track_id)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Cor baseada na classe
        class_name = proposal.get('suggested_class', 'comportamento_normal')
        color = self.CLASS_COLORS.get(class_name, (0, 255, 0))
        
        # Desenhar bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Label
        track_id = proposal.get('track_id', '?')
        conf = proposal.get('event_characteristics', {}).get('confidence_avg', 0)
        if conf == 0:
            conf = proposal.get('confidence', 0)
        
        label = f"Track {track_id} ({conf:.2f})"
        
        # Background para texto
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, 
            (x1, y1 - label_h - 10), 
            (x1 + label_w + 10, y1),
            color,
            -1
        )
        
        # Texto
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    def _display_frame(self, frame):
        """Converte frame OpenCV para Tkinter e exibe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize para caber na tela
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 1100, 650
        
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=img)
        
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Manter referencia
    
    def _show_no_image_message(self, message="Sem preview disponivel"):
        """Exibe mensagem quando nao ha imagem"""
        # Criar imagem preta com texto
        img = Image.new('RGB', (800, 600), color='black')
        photo = ImageTk.PhotoImage(image=img)
        
        self.video_label.config(image=photo, text=message, compound=tk.CENTER, fg='white')
        self.video_label.image = photo
    
    def _get_chunk_path(self, proposal: Dict) -> Path:
        """Obtem caminho do chunk a partir do proposal"""
        # Tentar obter chunk_id direto do proposal (formato do proposals_metadata.json)
        chunk_id = proposal.get('chunk_id', '')
        
        # Se nao tiver, tentar metadata
        if not chunk_id:
            metadata = proposal.get('metadata', {})
            chunk_id = metadata.get('chunk_id', '')
        
        # Se ainda nao tiver, tentar extrair do event_id
        if not chunk_id:
            event_id = proposal.get('event_id', '')
            if '_track_' in event_id:
                chunk_id = event_id.split('_track_')[0]
            elif 'event_' in event_id:
                # Formato: event_0001 -> chunk_0000 (inferir)
                # Extrair numero e tentar mapear
                try:
                    event_num = int(event_id.replace('event_', ''))
                    # Assumir que cada chunk tem ~10-20 eventos
                    chunk_num = event_num // 15
                    chunk_id = f"chunk_{chunk_num:04d}"
                except:
                    chunk_id = "chunk_0000"
        
        # Buscar arquivo .mp4
        chunk_path = self.chunks_dir / f"{chunk_id}.mp4"
        
        # Se nao encontrar, tentar sem zeros (chunk_0 em vez de chunk_0000)
        if not chunk_path.exists():
            # Extrair numero do chunk
            try:
                chunk_num = int(chunk_id.replace('chunk_', ''))
                alternate_path = self.chunks_dir / f"chunk_{chunk_num}.mp4"
                if alternate_path.exists():
                    return alternate_path
            except:
                pass
        
        return chunk_path
    
    def play_video(self):
        """Reproduzir video do evento completo com bboxes"""
        if self.playing:
            self.stop_playback()
            return
        
        proposal = self.proposals[self.current_idx]
        chunk_path = self._get_chunk_path(proposal)
        
        if not chunk_path.exists():
            messagebox.showerror("Erro", f"Video nao encontrado:\n{chunk_path}")
            return
        
        # Obter range de frames (pode estar em metadata ou event_characteristics)
        metadata = proposal.get('metadata', {})
        event_chars = proposal.get('event_characteristics', {})
        
        start_frame = event_chars.get('start_frame') or metadata.get('start_frame', 0)
        end_frame = event_chars.get('end_frame') or metadata.get('end_frame', 0)
        
        # Se nao tiver frames definidos, usar bbox_sequence
        bbox_sequence = proposal.get('bbox_sequence', [])
        if not start_frame and not end_frame and bbox_sequence:
            # Assumir que bbox_sequence[0] = frame start_frame
            start_frame = 0
            end_frame = len(bbox_sequence) - 1
        
        if start_frame == end_frame:
            messagebox.showwarning("Aviso", "Evento muito curto para playback. Use o frame estatico.")
            return
        
        self.playing = True
        self.play_button.config(text="⏸ Pausar (P)", bg='#FF5722')
        
        # Abrir video
        cap = cv2.VideoCapture(str(chunk_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = int(1000 / fps)
        
        # Tentar usar annotations (formato CVAT) ou bbox_sequence (formato proposals)
        annotations = proposal.get('annotations', [])
        use_bbox_sequence = not annotations and bbox_sequence
        
        class_name = proposal.get('suggested_class', 'comportamento_normal')
        color = self.CLASS_COLORS.get(class_name, (0, 255, 0))
        track_id = proposal.get('track_id', '?')
        
        frame_counter = [start_frame]  # Usar lista para manter referencia mutavel
        
        def play_frame():
            if not self.playing:
                cap.release()
                return
            
            ret, frame = cap.read()
            current_frame = frame_counter[0]
            frame_counter[0] += 1
            
            if not ret or current_frame > end_frame:
                self.stop_playback()
                cap.release()
                return
            
            # Desenhar bbox
            bbox = None
            
            if use_bbox_sequence:
                # Formato proposals_metadata.json: bbox_sequence direto
                bbox_idx = current_frame - start_frame
                if 0 <= bbox_idx < len(bbox_sequence):
                    bbox = bbox_sequence[bbox_idx]
            else:
                # Formato CVAT: annotations com image_id
                ann = next(
                    (a for a in annotations if a.get('image_id') == current_frame),
                    None
                )
                if ann:
                    bbox = ann.get('bbox', [])
            
            # Desenhar bbox se disponivel
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Label
                conf = event_chars.get('confidence_avg', 0)
                if conf == 0:
                    conf = proposal.get('confidence', 0)
                
                label = f"Track {track_id}"
                if conf > 0:
                    label += f" ({conf:.2f})"
                
                # Background do texto
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_h - 10), 
                    (x1 + label_w + 10, y1),
                    color,
                    -1
                )
                
                # Texto
                cv2.putText(
                    frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
            
            # Exibir frame
            self._display_frame(frame)
            
            # Proximo frame
            self.root.after(delay, play_frame)
        
        # Iniciar playback
        play_frame()
    
    def stop_playback(self):
        """Para playback do video"""
        self.playing = False
        self.play_button.config(text="▶ Play Video (P)", bg='#2196F3')
        
        # Recarregar frame estatico
        if 0 <= self.current_idx < len(self.proposals):
            self._load_representative_frame(self.proposals[self.current_idx])
    
    def approve(self):
        """Aprovar proposta"""
        proposal = self.proposals[self.current_idx].copy()
        proposal['status'] = 'approved'
        proposal['final_class'] = proposal.get('suggested_class')
        proposal['reviewed_at'] = datetime.now().isoformat()
        
        self.approved.append(proposal)
        logger.info(f"Proposta aprovada: {proposal.get('event_id')}")
        
        # Auto-save progresso
        self._save_progress()
        
        self.next_proposal()
    
    def reject(self):
        """Rejeitar proposta"""
        proposal = self.proposals[self.current_idx].copy()
        proposal['status'] = 'rejected'
        proposal['reviewed_at'] = datetime.now().isoformat()
        
        self.rejected.append(proposal)
        logger.info(f"Proposta rejeitada: {proposal.get('event_id')}")
        
        # Auto-save progresso
        self._save_progress()
        
        self.next_proposal()
    
    def correct_class(self):
        """Corrigir classe"""
        proposal = self.proposals[self.current_idx].copy()
        new_class = self.class_var.get()
        
        if new_class != proposal.get('suggested_class'):
            proposal['status'] = 'corrected'
            proposal['original_class'] = proposal.get('suggested_class')
            proposal['final_class'] = new_class
            proposal['reviewed_at'] = datetime.now().isoformat()
            
            self.corrected.append(proposal)
            logger.info(
                f"Proposta corrigida: {proposal.get('event_id')} "
                f"({proposal['original_class']} -> {new_class})"
            )
        else:
            # Sem mudanca, aprovar
            proposal['status'] = 'approved'
            proposal['final_class'] = new_class
            proposal['reviewed_at'] = datetime.now().isoformat()
            self.approved.append(proposal)
        
        # Auto-save progresso
        self._save_progress()
        
        self.next_proposal()
    
    def next_proposal(self):
        """Proxima proposta"""
        if self.current_idx < len(self.proposals) - 1:
            self.stop_playback()
            self.load_proposal(self.current_idx + 1)
        else:
            # Ultima proposta
            response = messagebox.askyesno(
                "Fim das Propostas",
                "Todas as propostas foram revisadas!\n\nDeseja finalizar e exportar resultados?"
            )
            if response:
                self.finish_review()
    
    def previous_proposal(self):
        """Proposta anterior"""
        if self.current_idx > 0:
            self.stop_playback()
            self.load_proposal(self.current_idx - 1)
    
    def _update_stats(self):
        """Atualiza estatisticas no footer"""
        stats_text = (
            f"Aprovadas: {len(self.approved)} | "
            f"Rejeitadas: {len(self.rejected)} | "
            f"Corrigidas: {len(self.corrected)}"
        )
        self.stats_label.config(text=stats_text)
    
    def finish_review(self):
        """Finalizar revisao e salvar resultados"""
        # Verificar se ha propostas nao revisadas
        total_reviewed = len(self.approved) + len(self.rejected) + len(self.corrected)
        
        if total_reviewed < len(self.proposals):
            remaining = len(self.proposals) - total_reviewed
            response = messagebox.askyesnocancel(
                "Revisao Incompleta",
                f"Ainda faltam {remaining} propostas para revisar.\n\n"
                f"Deseja:\n"
                f"• SIM: Exportar apenas as revisadas\n"
                f"• NAO: Continuar revisando\n"
                f"• CANCELAR: Sair sem salvar"
            )
            
            if response is None:  # Cancelar
                return
            elif response is False:  # Nao (continuar)
                return
            # Se Yes, continuar para export
        
        # Criar resultados
        results = {
            'review_session': {
                'start_time': datetime.now().isoformat(),
                'proposals_file': str(self.proposals_path),
                'total_proposals': len(self.proposals),
                'reviewed': total_reviewed,
                'pending': len(self.proposals) - total_reviewed
            },
            'approved': self.approved,
            'rejected': self.rejected,
            'corrected': self.corrected,
            'summary': {
                'total': len(self.proposals),
                'approved': len(self.approved),
                'rejected': len(self.rejected),
                'corrected': len(self.corrected),
                'approval_rate': len(self.approved) / len(self.proposals) if self.proposals else 0,
                'correction_rate': len(self.corrected) / len(self.proposals) if self.proposals else 0
            }
        }
        
        # Salvar resultados
        output_dir = Path('data_processing/annotations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'review_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados salvos em: {output_path}")
        
        # Exportar para YOLO format
        try:
            export_path = self._export_yolo_format(results)
            logger.info(f"Dataset YOLO exportado para: {export_path}")
            export_msg = f"\n\nDataset YOLO exportado para:\n{export_path}"
        except Exception as e:
            logger.error(f"Erro ao exportar YOLO: {e}")
            export_msg = f"\n\nErro ao exportar YOLO:\n{str(e)}"
        
        # Mensagem de conclusao
        summary = results['summary']
        msg = f"""
╔═══════════════════════════════════════════╗
║         REVISAO CONCLUIDA COM SUCESSO!    ║
╚═══════════════════════════════════════════╝

📊 ESTATISTICAS:
  Total de propostas: {summary['total']}
  Aprovadas: {summary['approved']}
  Rejeitadas: {summary['rejected']}
  Corrigidas: {summary['corrected']}

📈 TAXAS:
  Aprovacao: {summary['approval_rate']*100:.1f}%
  Correcao: {summary['correction_rate']*100:.1f}%

💾 ARQUIVOS GERADOS:
  Resultados: {output_path}{export_msg}
        """
        
        messagebox.showinfo("Revisao Concluida", msg)
        logger.info("Revisao finalizada com sucesso")
        
        # Remover arquivo de progresso apos finalizacao
        self._delete_progress()
        
        self.root.destroy()
    
    def _export_yolo_format(self, results: Dict) -> Path:
        """
        Exporta anotacoes aprovadas para formato YOLO
        Cria estrutura yolo_dataset/ com train/val/test splits
        """
        # Coletar apenas aprovadas e corrigidas
        valid_annotations = results['approved'] + results['corrected']
        
        if not valid_annotations:
            raise ValueError("Nenhuma anotacao valida para exportar")
        
        # Criar estrutura
        output_base = Path('yolo_dataset_reviewed')
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Criar splits (70/15/15)
        import random
        random.shuffle(valid_annotations)
        
        n = len(valid_annotations)
        train_split = int(n * 0.7)
        val_split = int(n * 0.85)
        
        splits = {
            'train': valid_annotations[:train_split],
            'val': valid_annotations[train_split:val_split],
            'test': valid_annotations[val_split:]
        }
        
        # Criar diretorios
        for split_name in ['train', 'val', 'test']:
            (output_base / split_name / 'images').mkdir(parents=True, exist_ok=True)
            (output_base / split_name / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Mapear classes para IDs
        class_to_id = {cls: i for i, cls in enumerate(self.CLASSES)}
        
        # Exportar cada anotacao
        for split_name, annotations in splits.items():
            for ann in annotations:
                self._export_annotation_to_yolo(
                    ann,
                    output_base / split_name,
                    class_to_id
                )
        
        # Criar dataset.yaml
        yaml_content = f"""# Dataset de Furtos - Revisado
path: {output_base.absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(self.CLASSES)}
names: {self.CLASSES}

# Estatisticas da revisao
stats:
  total_annotations: {len(valid_annotations)}
  train_size: {len(splits['train'])}
  val_size: {len(splits['val'])}
  test_size: {len(splits['test'])}
  approval_rate: {results['summary']['approval_rate']:.2%}
"""
        
        with open(output_base / 'dataset.yaml', 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"Exportados {len(valid_annotations)} eventos para YOLO format")
        
        return output_base
    
    def _export_annotation_to_yolo(
        self,
        annotation: Dict,
        output_dir: Path,
        class_to_id: Dict[str, int]
    ):
        """Exporta uma anotacao individual para YOLO format"""
        # Obter classe final
        final_class = annotation.get('final_class', annotation.get('suggested_class'))
        class_id = class_to_id.get(final_class, 0)
        
        # Obter caminho do chunk
        chunk_path = self._get_chunk_path(annotation)
        
        if not chunk_path.exists():
            logger.warning(f"Chunk nao encontrado para export: {chunk_path}")
            return
        
        # Processar cada frame
        annotations_list = annotation.get('annotations', [])
        images_list = annotation.get('images', [])
        
        cap = cv2.VideoCapture(str(chunk_path))
        
        for img_info, ann_info in zip(images_list, annotations_list):
            frame_id = img_info['id']
            
            # Ler frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Salvar imagem
            event_id = annotation.get('event_id', 'unknown')
            image_filename = f"{event_id}_frame_{frame_id:06d}.jpg"
            image_path = output_dir / 'images' / image_filename
            cv2.imwrite(str(image_path), frame)
            
            # Salvar label YOLO format
            bbox = ann_info.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                img_h, img_w = img_info['height'], img_info['width']
                
                # Converter para YOLO format (x_center, y_center, width, height) normalizado
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                # Salvar label
                label_filename = f"{event_id}_frame_{frame_id:06d}.txt"
                label_path = output_dir / 'labels' / label_filename
                
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        cap.release()
    
    def on_closing(self):
        """Handler para fechamento da janela"""
        total_reviewed = len(self.approved) + len(self.rejected) + len(self.corrected)
        
        if total_reviewed > 0:
            response = messagebox.askyesnocancel(
                "Sair",
                f"Progresso sera salvo automaticamente.\n\n"
                f"Revisadas: {total_reviewed}/{len(self.proposals)}\n"
                f"Aprovadas: {len(self.approved)}\n"
                f"Rejeitadas: {len(self.rejected)}\n"
                f"Corrigidas: {len(self.corrected)}\n\n"
                f"Deseja:\n"
                f"• SIM: Salvar progresso e sair\n"
                f"• NAO: Sair sem salvar (perder progresso)\n"
                f"• CANCELAR: Continuar revisando"
            )
            
            if response is None:  # Cancelar
                return
            elif response is True:  # Sim (salvar e sair)
                self._save_progress()
                logger.info(f"Progresso salvo. Voce pode continuar depois executando novamente a GUI.")
            elif response is False:  # Nao (sair sem salvar)
                logger.warning("Saindo sem salvar progresso. Progresso sera perdido.")
        
        self.stop_playback()
        self.root.destroy()
    
    def run(self):
        """Iniciar interface"""
        logger.info("Iniciando interface de revisao")
        self.root.mainloop()


def main():
    """Funcao principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interface de Revisao Humana - Pipeline Automatizado'
    )
    parser.add_argument(
        '--proposals',
        type=str,
        default='data_processing/proposals/proposals_metadata.json',
        help='Caminho para arquivo de propostas'
    )
    parser.add_argument(
        '--chunks',
        type=str,
        default='data_processing/active_chunks',
        help='Diretorio com chunks de video'
    )
    
    args = parser.parse_args()
    
    # Verificar se arquivo existe
    if not os.path.exists(args.proposals):
        print(f"❌ Erro: Arquivo nao encontrado: {args.proposals}")
        print("\nCrie propostas primeiro executando o pipeline ate o estagio de labeling.")
        return 1
    
    try:
        gui = ProposalReviewGUI(args.proposals, args.chunks)
        gui.run()
        return 0
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        messagebox.showerror("Erro Fatal", f"Erro ao iniciar interface:\n\n{str(e)}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

#    __  ____ ____ _  _ 
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
