# 🎮 Jogo de Memória e Coordenação com Visão Computacional

Um webapp interativo que combina memória visual e coordenação motora usando visão computacional (OpenCV) e MediaPipe para detectar movimentos das mãos em tempo real. O jogador deve memorizar sequências de cores e reproduzi-las usando gestos das mãos em áreas específicas da tela.

## 🎯 Como Funciona

### Fase 1: Memorização

- O webapp exibe a câmera do usuário em tempo real
- Uma sequência de cores pisca por toda a borda da tela
- O jogador deve memorizar a ordem das cores apresentadas

### Fase 2: Execução

- A tela tem circulos coloridos espalhados;
- Indicações mostram qual **mão** (esquerda/direita) deve ser usada
- O jogador deve tocar as áreas na sequência correta usando a mão indicada

### Progressão

- ✅ **Sucesso**: Nova sequência mais longa é gerada
- ❌ **Erro**: Jogo reinicia ou dá nova chance
- 🏆 **Objetivo**: Alcançar sequências cada vez mais longas

## 🚀 Recursos

- **Detecção de mãos em tempo real** com MediaPipe
- **Interface responsiva** com pygame/OpenCV
- **Sistema de pontuação progressiva**
- **Randomização** de cores e indicações de mãos
- **Gamificação** com dificuldade crescente
- **Feedback visual** em tempo real

## 📋 Pré-requisitos

- Python 3.12+
- Câmera web funcional
- Boa iluminação para detecção de mãos
- `uv` instalado (gerenciador de pacotes Python)

## 🚀 Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/MathBorgess/multimedia-prog.git
   cd multimedia-prog
   ```

2. Instale as dependências com `uv`:
   ```bash
   uv sync
   ```

## ▶️ Como Jogar

1. **Teste a câmera** (recomendado):

   ```bash
   python test_camera.py
   ```

2. **Execute o jogo**:

   ```bash
   python __main__.py
   ```

3. **Posicione-se** na frente da câmera com boa iluminação
4. **Memorize** a sequência de cores que pisca na borda
5. **Toque** as áreas coloridas com a mão indicada na ordem correta
6. **Avance** para sequências mais longas e desafiadoras!

## 🎛️ Controles

- **Movimentos das mãos**: Interagir com as áreas do jogo
- **Tecla 'q'**: Sair do programa
- **Gestos detectados**: Posição das mãos em tempo real

## 🛠️ Estrutura do Projeto

```
multimedia-prog/
├── __main__.py            # Ponto de entrada principal
├── test_camera.py         # Script de teste de câmera
├── TROUBLESHOOTING.md     # Guia de solução de problemas
├── README.md              # Este arquivo
├── requirements.txt       # Dependências do projeto
├── pyproject.toml         # Configuração do projeto
├── uv.lock               # Lock file do uv
├── config/               # Configurações do jogo
│   └── config.py         # Parâmetros e configurações
├── playground/           # Módulos principais do jogo
│   ├── virtual_drums.py  # Engine principal (será refatorado)
│   ├── drum_kit.py      # Componentes do jogo (será refatorado)
│   └── drum.py          # Utilitários
└── sounds/              # Assets de áudio
    ├── crash_1.wav
    ├── hihat_1.wav
    └── snare_1.wav
```

## 🎨 Fluxo do Jogo

1. **Inicialização**: Detecção e configuração da câmera
2. **Sequência de Cores**: Exibição da sequência a ser memorizada
3. **Tela Interativa**: Circulos coloridos sobre a tela randomicamente
4. **Detecção de Gestos**: Rastreamento das mãos do jogador
5. **Validação**: Verificação da sequência executada
6. **Progressão**: Aumento da dificuldade com sequências mais longas

## 🔧 Configurações

O arquivo `config/config.py` permite ajustar:

- Sensibilidade de detecção das mãos
- Tempo de exibição das sequências
- Cores utilizadas no jogo
- Configurações da câmera
- Parâmetros de dificuldade

## 🚨 Solução de Problemas

Se encontrar problemas com a câmera, consulte o arquivo `TROUBLESHOOTING.md` que contém:

- Soluções para problemas comuns de câmera
- Instruções para permissões no macOS
- Script de teste independente
- Dicas de performance

## 🤝 Contribuição

Contribuições são bem-vindas! Este projeto está em desenvolvimento ativo. Áreas que precisam de desenvolvimento:

- **Interface do Jogo**: Implementação da randomização dos circulos na tela
- **Sistema de Pontuação**: Tracking de scores e níveis
- **Efeitos Visuais**: Melhor feedback visual para interações
- **Sistema de Cores**: Otimização da sequência de cores
- **Performance**: Otimização da detecção de gestos

Sinta-se à vontade para abrir issues e enviar pull requests!

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
