"""
osu! Memory Reader
Reads game state and timing directly from osu! process memory
"""
import pymem
import pymem.process
from typing import Optional
import time


class OsuMemoryReader:
    """
    Reads osu! game state from memory

    Memory structure (as of current osu! versions):
    - Base address found via signature scanning
    - Offsets to important values (audio time, game state, etc.)
    """

    def __init__(self):
        self.pm: Optional[pymem.Pymem] = None
        self.base_address: Optional[int] = None
        self.connected = False

        # Memory offsets (these may need updating with osu! versions)
        # These are common offsets - verify with Cheat Engine if needed
        self.AUDIO_TIME_OFFSET = 0x46  # Offset to current audio time
        self.STATUS_OFFSET = 0x04      # Offset to game status

    def connect(self) -> bool:
        """
        Connect to osu! process
        Returns True if successful, False otherwise
        """
        try:
            self.pm = pymem.Pymem("osu!.exe")
            print("Connected to osu! process")

            # Find base address via signature scan
            self.base_address = self._find_base_address()

            if self.base_address:
                print(f"Found base address: 0x{self.base_address:X}")
                self.connected = True
                return True
            else:
                print("Could not find base address")
                return False

        except pymem.exception.ProcessNotFound:
            print("osu! is not running")
            return False
        except Exception as e:
            print(f"Error connecting to osu!: {e}")
            return False

    def _find_base_address(self) -> Optional[int]:
        """
        Find base address using signature scanning
        This searches for a known pattern in memory
        """
        try:
            # Pattern for osu! base (this is a simplified approach)
            # In production, use a more robust signature scan

            # Get the base module (osu!.exe)
            module = pymem.process.module_from_name(self.pm.process_handle, "osu!.exe")

            if module:
                # For simplicity, return module base
                # In production, you'd scan for specific patterns
                return module.lpBaseOfDll

            return None

        except Exception as e:
            print(f"Error finding base address: {e}")
            return None

    def get_audio_time(self) -> Optional[int]:
        """
        Get current audio time in milliseconds
        Returns None if not available
        """
        if not self.connected or not self.base_address:
            return None

        try:
            # This is a simplified read - actual implementation needs proper offset chain
            # The real offset chain looks like: [[base + offset1] + offset2] + offset3...

            # Read pointer at base
            ptr = self.pm.read_int(self.base_address + 0x00579E2C)  # Example offset
            if ptr:
                # Read audio time from pointer + offset
                audio_time = self.pm.read_int(ptr + 0x46)
                return audio_time

            return None

        except Exception as e:
            # Silently fail - memory reading can be flaky
            return None

    def get_game_status(self) -> Optional[int]:
        """
        Get current game status

        Status codes:
        0 = Main menu
        1 = Editing
        2 = Playing
        7 = Watching replay
        etc.

        Returns None if not available
        """
        if not self.connected or not self.base_address:
            return None

        try:
            # Read game status from memory
            ptr = self.pm.read_int(self.base_address + 0x00579E2C)
            if ptr:
                status = self.pm.read_int(ptr + 0x04)
                return status

            return None

        except Exception as e:
            return None

    def is_playing(self) -> bool:
        """
        Check if currently playing a beatmap
        """
        status = self.get_game_status()
        return status == 2 if status is not None else False

    def wait_for_playing(self, timeout: float = 60.0) -> bool:
        """
        Wait until osu! starts playing a beatmap

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if playing started, False if timeout
        """
        start_time = time.time()

        print("Waiting for beatmap to start...")
        print("(Start playing in osu! now)")

        while time.time() - start_time < timeout:
            if self.is_playing():
                print("Beatmap started!")
                return True
            time.sleep(0.1)

        print("Timeout waiting for beatmap to start")
        return False

    def get_sync_time(self) -> Optional[int]:
        """
        Get the current audio time for syncing
        Returns time in milliseconds
        """
        return self.get_audio_time()

    def disconnect(self):
        """
        Disconnect from osu! process
        """
        if self.pm:
            self.pm.close_process()
            self.connected = False
            print("Disconnected from osu!")


# Simple memory offsets that work for most osu! versions
# These need to be updated if they break
OSU_MEMORY_OFFSETS = {
    # These are example offsets - they WILL need updating
    # Use Cheat Engine to find current offsets for your osu! version
    "base_pattern": b"\x8B\x0D....\x85\xC9\x74\x1C",  # Pattern to find base
    "audio_time_chain": [0x00579E2C, 0x46],  # Pointer chain to audio time
    "status_chain": [0x00579E2C, 0x04],      # Pointer chain to status
}
