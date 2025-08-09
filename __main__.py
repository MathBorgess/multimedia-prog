import asyncio
import platform
import logging
from playground.memory_game import MemoryGame
from config.config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main async function for Pyodide compatibility."""
    app = None
    try:
        app = MemoryGame()
        app.setup()
        logger.info("Memory Game started successfully!")

        while True:
            app.update_loop()
            await asyncio.sleep(0.1 / CONFIG['fps'])

    except KeyboardInterrupt:
        logger.info("Game interrupted by user (Ctrl+C)")
    except SystemExit:
        logger.info("Game exited normally")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        logger.error("Please check camera connection and try again")
        # Show user-friendly error message
        print("\n" + "="*50)
        print("ERROR: Memory Game encountered an issue")
        print(f"Details: {e}")
        print("\nPossible solutions:")
        print("1. Check if your camera is connected and working")
        print("2. Close other applications that might be using the camera")
        print("3. Try running the game again")
        print("4. Press 'q' to quit if the camera window is open")
        print("="*50 + "\n")
    finally:
        if app:
            try:
                app.cleanup()
                logger.info("Game cleanup completed")
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
