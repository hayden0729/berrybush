# standard imports
from datetime import datetime
from pathlib import Path
import shutil
# 3rd party imports
import bpy
# internal imports
from .preferences import getPrefs


class ClearBackups(bpy.types.Operator):
    """Clear the BerryBush backups directory."""

    bl_idname = "brres.clear_backups"
    bl_label = "Clear Backups"

    def execute(self, context: bpy.types.Context):
        backupDir = getPrefs(context).backupDir
        backupPath = Path(backupDir)
        for file in backupPath.iterdir():
            if isBackup(file):
                file.unlink()
        return {'FINISHED'}


def tryBackup(filepath: str, context: bpy.types.Context):
    """Back up a file to the BerryBush backups directory if backups are enabled and the file exists.

    If a backup is created and the max capacity is enabled and exceeded,
    the oldest file(s) in the directory are deleted until at capacity.
    """
    prefs = getPrefs(context)
    path = Path(filepath)
    if not prefs.doBackups or not path.exists():
        return
    # path is ensured to be absolute for the sake of clear error messages
    backupFolderPath = Path(prefs.backupDir).absolute()
    try:
        backupFolderPath.mkdir(exist_ok=True)
    except OSError as e:
        raise OSError("Failed to open the BerryBush backup folder. "
                      "You may have to change its location in the addon preferences, "
                      "or make sure the path to it exists. See above for more info.") from e
    backupLabel = datetime.now().strftime("(backup %Y-%m-%d %H-%M-%S)")
    backupPath = Path(backupFolderPath, f"{path.stem} {backupLabel}{path.suffix}")
    try:
        shutil.copy(filepath, str(backupPath))
    except OSError as e:
        raise OSError("Failed to create a BRRES backup. See above for more info.") from e
    if prefs.doMaxBackups:
        backupFiles = [f for f in backupFolderPath.iterdir() if isBackup(f)]
        backupFiles.sort(key=lambda file: file.stat().st_mtime, reverse=True)
        while len(backupFiles) > prefs.maxBackups:
            try:
                backupFiles.pop().unlink()
            except OSError as e:
                raise OSError("Failed to delete an old BRRES backup. "
                              "See above for more info.") from e


def isBackup(path: Path):
    """Whether a path should be considered a BerryBush backup."""
    return path.is_file() and path.suffix == ".brres" and "backup" in path.stem
