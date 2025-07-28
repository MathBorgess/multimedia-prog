#!/usr/bin/env python3
"""
Script de teste para verificar se a câmera e o MediaPipe estão funcionando corretamente.
"""

import cv2
import mediapipe as mp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_camera():
    """Testa a câmera básica."""
    logger.info("Testando câmera...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Não foi possível abrir a câmera.")
        return False

    ret, frame = cap.read()
    if not ret:
        logger.error("Não foi possível ler frame da câmera.")
        cap.release()
        return False

    h, w = frame.shape[:2]
    logger.info(f"Câmera funcionando: {w}x{h}")
    cap.release()
    return True


def test_mediapipe():
    """Testa o MediaPipe Hands."""
    logger.info("Testando MediaPipe Hands...")

    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        hands.close()
        logger.info("MediaPipe Hands funcionando!")
        return True
    except Exception as e:
        logger.error(f"Erro no MediaPipe: {e}")
        return False


def test_integration():
    """Testa a integração câmera + MediaPipe."""
    logger.info("Testando integração câmera + MediaPipe...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Câmera não disponível.")
        return False

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    try:
        logger.info("Pressione 'q' para sair do teste.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Falha ao ler frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Draw hand landmarks
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Add instructions
            cv2.putText(frame, "Mova suas maos na frente da camera", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Pressione 'q' para sair", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Teste de Integração', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Erro na integração: {e}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    logger.info("Teste de integração concluído!")
    return True


def main():
    """Executa todos os testes."""
    logger.info("🚀 Iniciando testes do sistema...")

    tests = [
        ("Câmera", test_camera),
        ("MediaPipe", test_mediapipe),
        ("Integração", test_integration)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 Teste: {test_name}")
        logger.info('='*50)

        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                logger.info(f"✅ {test_name}: PASSOU")
            else:
                logger.error(f"❌ {test_name}: FALHOU")

        except Exception as e:
            logger.error(f"❌ {test_name}: ERRO - {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 RESUMO DOS TESTES")
    logger.info('='*50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nResultado: {passed}/{total} testes passaram")

    if passed == total:
        logger.info("🎉 Todos os testes passaram! O sistema está pronto!")
    else:
        logger.warning(
            "⚠️  Alguns testes falharam. Verifique as configurações.")


if __name__ == "__main__":
    main()
