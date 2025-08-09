# 📋 Documentação Técnica - Jogo de Memória e Coordenação

## 📖 Visão Geral

Este projeto implementa um jogo interativo de memória e coordenação motora que utiliza visão computacional para detectar movimentos das mãos em tempo real. O sistema combina tecnologias de processamento de imagem (OpenCV), detecção de landmarks corporais (MediaPipe) e interface gráfica (Pygame) para criar uma experiência gamificada.

### 🎯 Objetivos do Sistema

- **Memória Visual**: Teste da capacidade de memorização de sequências de cores
- **Coordenação Motora**: Exercício de precisão e timing com gestos das mãos
- **Interação Natural**: Interface sem necessidade de dispositivos físicos além da câmera
- **Progressão Adaptativa**: Dificuldade crescente baseada no desempenho do usuário

## 🏗️ Arquitetura do Sistema

### 🗂️ Estrutura de Diretórios

```
multimedia-prog/
├── __main__.py              # Ponto de entrada principal
├── test_system.py           # Script de teste e diagnóstico
├── test_thumbs_up.py        # Teste específico para gestos
├── generate_sounds.py       # Gerador de assets de áudio
├── run_game.sh             # Script de execução Linux/macOS
├── pyproject.toml          # Configuração do projeto e dependências
├── requirements.txt        # Dependências Python
├── uv.lock                # Lock file do gerenciador uv
├── README.md              # Documentação do usuário
├── THUMBS_UP_GUIDE.md     # Guia de implementação de gestos
├── LICENSE                # Licença do projeto
├── config/                # Módulo de configurações
│   ├── config.py          # Parâmetros centralizados do sistema
│   └── __pycache__/       # Cache Python
├── playground/            # Módulos principais do jogo
│   ├── memory_game.py     # Classe principal do jogo
│   ├── virtual_drums.py   # Engine de áudio (legado)
│   ├── drum_kit.py       # Componentes de áudio (legado)
│   ├── drum.py           # Utilitários de áudio
│   └── __pycache__/      # Cache Python
└── sounds/               # Assets de áudio
    ├── success.wav       # Som de sucesso
    ├── error.wav         # Som de erro
    ├── sequence.wav      # Som da sequência
    ├── crash_1.wav       # Som de crash (drums)
    ├── hihat_1.wav       # Som de hihat (drums)
    └── snare_1.wav       # Som de snare (drums)
```

### 🏗️ Decisões Arquiteturais

### 🔧 Componentes Principais

#### 1. **Módulo Principal (`__main__.py`)**
- **Função**: Ponto de entrada e loop principal assíncrono
- **Responsabilidades**:
  - Inicialização do sistema de logging
  - Gerenciamento do loop principal do jogo
  - Compatibilidade com Pyodide (execução web)
  - Tratamento de exceções globais


#### 2. **Classe MemoryGame (`playground/memory_game.py`)**
- **Função**: Core do sistema de jogo
- **Responsabilidades**:
  - Gerenciamento de estados do jogo
  - Detecção e processamento de mãos via MediaPipe
  - Lógica de sequências e validação
  - Renderização da interface gráfica
  - Integração com sistema de áudio

**Estados do Jogo:**
- `INIT`: Inicialização
- `SHOW_SEQUENCE`: Exibição da sequência a memorizar
- `WAIT_INPUT`: Aguardando entrada do usuário
- `CHECKING`: Validando entrada
- `SUCCESS`: Sucesso na sequência
- `FAILURE`: Falha na sequência

#### 3. **Sistema de Configuração (`config/config.py`)**
- **Função**: Centralização de parâmetros configuráveis
- **Categorias**:
  - **Game Settings**: Duração de sequências, pausas, comprimento inicial
  - **Visual Settings**: Transparência, espessura de bordas
  - **Sound Settings**: Habilitação e arquivos de áudio
  - **Hand Detection**: Thresholds de detecção e sensibilidade
  - **Camera Settings**: Resolução, FPS, índice da câmera
  - **MediaPipe Config**: Configurações do modelo de detecção de mãos
  - **Performance**: FPS do jogo
  - **Game Colors**: Paleta de cores RGB

## 🛠️ Tecnologias Utilizadas

### 📚 Dependências Principais

| Biblioteca | Versão | Função |
|------------|---------|---------|
| **MediaPipe** | ≥0.10.21 | Detecção de landmarks das mãos |
| **OpenCV** | ≥4.11.0.86 | Processamento de imagem e vídeo |
| **NumPy** | ≥1.26.4 | Operações matemáticas e arrays |
| **Pygame** | ≥2.6.1 | Interface gráfica e áudio |


### 🔧 Escolha de Ferramentas

#### **MediaPipe**

**Por que MediaPipe?**
- ✅ **Performance Superior**: Otimizado para tempo real
- ✅ **Precisão**: 21 landmarks por mão com alta acurácia
- ✅ **Facilidade de Uso**: API Python simples e bem documentada
- ✅ **Suporte Multiplataforma**: Funciona em Windows, macOS, Linux
- ✅ **Otimização Mobile**: Preparado para deploy em dispositivos móveis


#### **OpenCV vs Alternativas**

**Por que OpenCV?**
- ✅ **Maturidade**: Biblioteca consolidada e estável
- ✅ **Performance**: Otimizada em C++ com bindings Python
- ✅ **Integração**: Funciona nativamente com MediaPipe
- ✅ **Funcionalidades**: Amplo conjunto de ferramentas de visão computacional
- ✅ **Comunidade**: Grande base de usuários e documentação


#### **Pygame vs Alternativas**

**Por que Pygame?**
- ✅ **Simplicidade**: API intuitiva para áudio e gráficos
- ✅ **Integração**: Funciona bem com OpenCV para display
- ✅ **Áudio**: Sistema de mixer robusto
- ✅ **Portabilidade**: Multiplataforma sem configuração adicional
- ✅ **Leveza**: Não adiciona overhead significativo

## 🚧 Desafios Encontrados e Soluções

### 🎯 Desafio 1: Precisão da Detecção de Mãos

**Problema:**
- Detecções inconsistentes em diferentes condições de iluminação
- Falsos positivos em movimentos rápidos
- Latência na detecção afetando a experiência

**Soluções Implementadas:**

1. **Otimização de Parâmetros MediaPipe:**
```python
'hands_config': {
    'max_num_hands': 2,
    'min_detection_confidence': 0.7,    # Aumentado de 0.5
    'min_tracking_confidence': 0.7      # Otimizado para estabilidade
}
```

2. **Sistema de Cooldown:**
```python
def check_interaction_cooldown(self):
    """Previne múltiplas detecções da mesma interação."""
    current_time = time.time()
    if current_time - self.last_interaction_time < self.interaction_cooldown:
        return False
    return True
```

3. **Algoritmo de Centro da Palma Melhorado:**
```python
def get_stable_hand_center(self, landmarks, width, height):
    """Usa múltiplos landmarks para maior estabilidade."""
    # Landmarks da palma (0, 5, 9, 13, 17) para maior precisão
    palm_points = [landmarks.landmark[i] for i in [0, 5, 9, 13, 17]]
    # Média ponderada para reduzir jitter
    return calculate_weighted_center(palm_points, width, height)
```

### 🎯 Desafio 2: Performance em Tempo Real

**Problema:**
- FPS baixo em hardware menos potente
- Latência entre detecção e resposta
- Uso intensivo de CPU

**Soluções Implementadas:**

1. **Otimização do Loop Principal:**
```python
async def main():
    """Loop assíncrono para melhor performance."""
    while True:
        app.update_loop()
        await asyncio.sleep(0.1 / CONFIG['fps'])  # Controle preciso de FPS
```

2. **Processamento Seletivo:**
```python
def selective_processing(self, frame):
    """Processa apenas quando necessário."""
    if self.game_state in ["WAIT_INPUT", "CHECKING"]:
        return self.process_hands(frame)
    return None  # Pula processamento em estados desnecessários
```

3. **Cache de Objetos MediaPipe:**
```python
# Reutilização de objetos para evitar overhead de criação
self.hands = self.mp_hands.Hands(**CONFIG['hands_config'])
# Mantém instância durante toda a execução
```

### 🎯 Desafio 4: Experiência do Usuário

**Problema:**
- Feedback visual insuficiente
- Curva de aprendizado steep
- Falta de guidance para posicionamento

**Soluções Implementadas:**

1. **Sistema de Feedback Visual:**
```python
def draw_interaction_feedback(self, frame, hand_pos, area_index):
    """Feedback visual para interações."""
    # Círculo verde para detecção válida
    cv2.circle(frame, hand_pos, 20, (0, 255, 0), 3)
    # Highlight da área ativa
    self.highlight_active_area(frame, area_index)
```

2. **Sistema de Instruções Dinâmicas:**
```python
def draw_instructions(self, frame):
    """Instruções contextuais baseadas no estado."""
    instructions = {
        "SHOW_SEQUENCE": "Memorize a sequência de cores",
        "WAIT_INPUT": "Toque as áreas com a mão indicada",
        "SUCCESS": "Parabéns! Próxima sequência...",
        "FAILURE": "Tente novamente!"
    }
    cv2.putText(frame, instructions[self.game_state], ...)
```

3. **Script de Teste Dedicado:**
```python
# test_system.py
def test_camera_and_hands():
    """Teste interativo para verificar setup."""
    # Permite ao usuário verificar se tudo está funcionando
    # antes de jogar
```

### 🎯 Desafio 5: Gerenciamento de Estado Complexo

**Problema:**
- Transições de estado inconsistentes
- Race conditions entre detecção e lógica
- Estado global compartilhado

**Soluções Implementadas:**

1. **State Machine Formal:**
```python
class GameStateMachine:
    """Máquina de estados formal com validação."""
    
    VALID_TRANSITIONS = {
        "INIT": ["SHOW_SEQUENCE"],
        "SHOW_SEQUENCE": ["WAIT_INPUT"],
        # ... outras transições
    }
    
    def transition_to(self, new_state):
        """Transição validada entre estados."""
        if new_state not in self.VALID_TRANSITIONS[self.current_state]:
            raise InvalidTransitionError(f"Cannot transition from {self.current_state} to {new_state}")
        self.current_state = new_state
```

2. **Locks para Race Conditions:**
```python
import threading

class MemoryGame:
    def __init__(self):
        self.state_lock = threading.Lock()
    
    def safe_state_transition(self, new_state):
        """Transição thread-safe."""
        with self.state_lock:
            self.game_state = new_state
```

3. **Event-Driven Architecture:**
```python
def handle_hand_interaction(self, hand_data):
    """Processa interação apenas no estado correto."""
    if self.game_state != "WAIT_INPUT":
        return  # Ignora interações em estados inadequados
    
    self.process_interaction(hand_data)
```

## 🎛️ Sistema de Configuração

### ⚙️ Parâmetros Principais

**Game Settings:**
```python
'sequence_start_length': 2,        # Comprimento inicial da sequência
'border_flash_duration': 1.0,      # Duração do flash da borda (segundos)
'color_pause_duration': 0.8,       # Pausa entre cores (segundos)
'success_pause_duration': 1.5,     # Pausa após sucesso (segundos)
'failure_pause_duration': 2.5,     # Pausa após falha (segundos)
```

**Visual Settings:**
```python
'area_transparency': 0.6,          # Transparência das áreas (0.0-1.0)
'border_thickness': 25,            # Espessura da borda para sequência
```

**Performance Settings:**
```python
'fps': 60,                         # FPS do jogo
'camera_fps': 30,                  # FPS da câmera
'camera_width': 640,               # Largura da câmera
'camera_height': 480,              # Altura da câmera
```

## 🚀 Performance e Otimização

### ⚡ Métricas de Performance

**Target FPS:**
- Jogo: 60 FPS
- Câmera: 30 FPS
- Processamento: ~16.67ms por frame

**Otimizações Implementadas:**
1. **Assíncrono**: Loop principal não-bloqueante
2. **Cache**: Reutilização de objetos MediaPipe
3. **Cooldown**: Prevenção de spam de detecções
4. **Processamento Seletivo**: Apenas quando necessário

### 📊 Monitoramento

```python
# Logging configurado para monitoramento
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

## 🔒 Tratamento de Erros

### 🛡️ Estratégias de Resiliência

1. **Try-Catch Globais**: Captura de exceções não tratadas
2. **Fallbacks**: Valores padrão para configurações
3. **Cleanup**: Liberação adequada de recursos
4. **Logging Detalhado**: Rastreamento de erros

### ⚠️ Cenários de Erro

| Erro | Causa | Tratamento |
|------|--------|------------|
| Câmera não disponível | Hardware/permissões | Fallback para câmera padrão |
| MediaPipe falha | Instalação/versão | Log e continuação sem detecção |
| Áudio não funciona | Sistema/arquivos | Modo silencioso |
| Performance baixa | Hardware limitado | Redução automática de qualidade |

## 💻 Análise Detalhada do Código

### 🔍 Estrutura da Classe Principal (MemoryGame)

A classe `MemoryGame` é o núcleo do sistema e implementa o padrão State Machine:

```python
class MemoryGame:
    def __init__(self):
        # Inicialização do MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Estado do jogo
        self.game_state = "INIT"
        self.sequence = []
        self.current_sequence_index = 0
        
        # Áreas do jogo (grid 3x3)
        self.game_areas = []
        self.area_colors = []
        self.area_hands = []
```

**Principais Métodos:**

1. **`setup()`**: Inicialização de recursos
   - Configuração da câmera
   - Inicialização do MediaPipe
   - Setup do Pygame para áudio

2. **`update_loop()`**: Loop principal do jogo
   - Captura de frames da câmera
   - Processamento de detecção de mãos
   - Atualização da lógica de estado
   - Renderização da interface

3. **`process_hands()`**: Processamento das mãos detectadas
   - Conversão de landmarks para coordenadas da tela
   - Detecção de interações com as áreas do jogo
   - Aplicação de cooldown para evitar spam

4. **`check_hand_interaction()`**: Validação de interações
   - Cálculo de proximidade entre mão e área
   - Verificação da mão correta (esquerda/direita)
   - Trigger de eventos de jogo

### 🎯 Sistema de Estados Detalhado

```python
def update_game_logic(self):
    """Atualiza a lógica baseada no estado atual."""
    if self.game_state == "INIT":
        self.initialize_new_game()
    elif self.game_state == "SHOW_SEQUENCE":
        self.display_sequence()
    elif self.game_state == "WAIT_INPUT":
        self.wait_for_user_input()
    elif self.game_state == "CHECKING":
        self.validate_user_input()
    elif self.game_state == "SUCCESS":
        self.handle_success()
    elif self.game_state == "FAILURE":
        self.handle_failure()
```

### 🖐️ Algoritmo de Detecção de Mãos

```python
def get_hand_center(self, landmarks, width, height):
    """Calcula o centro da palma da mão."""
    # Usa landmarks específicos da palma (índices 0, 5, 9, 13, 17)
    palm_landmarks = [0, 5, 9, 13, 17]
    x_coords = [landmarks.landmark[i].x * width for i in palm_landmarks]
    y_coords = [landmarks.landmark[i].y * height for i in palm_landmarks]
    
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    
    return int(center_x), int(center_y)
```

## 👥 Participação dos Membros da Equipe

### 🏗️ Divisão de Responsabilidades

### 🤝 Metodologia de Desenvolvimento

**Workflow Colaborativo:**
1. **Todo**: Tarefas a serem executadas
2. **Doing**: Desenvolvimento paralelo de módulos
3. **Code Review**: Revisão cruzada de código
4. **Testing**: Testes individuais e de integração
5. **Done**: Tarefa concluida

**Ferramentas de Colaboração:**
- **Git**: Controle de versão distribuído
- **GitHub**: Repositório central e issue tracking
- **Code Review**: Revisão cruzada de código

