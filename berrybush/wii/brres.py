# standard imports
from functools import cache
from struct import Struct
from typing import TypeVar
# internal imports
from .binaryutils import pad
from .brresdict import DictReader, DictWriter
from .serialization import Serializer, Reader, Writer, StrPoolReadMixin, StrPoolWriteMixin
from .subfile import Subfile, SubfileReader, SubfileWriter
from .common import getKey
from .mdl0 import MDL0, MDL0Reader, MDL0Writer
from .tex0 import TEX0, TEX0Reader, TEX0Writer
from .plt0 import PLT0, PLT0Reader, PLT0Writer
from .chr0 import CHR0, CHR0Reader, CHR0Writer
from .clr0 import CLR0, CLR0Reader, CLR0Writer
from .pat0 import PAT0, PAT0Reader, PAT0Writer
from .srt0 import SRT0, SRT0Reader, SRT0Writer
from .vis0 import VIS0, VIS0Reader, VIS0Writer


SUBFILE_T = TypeVar("SUBFILE_T", bound=Subfile)
SUBFILE_TYPES = (MDL0, TEX0, PLT0, CHR0, CLR0, PAT0, SRT0, VIS0)


class BRRES():
    """Binary Revolution RESource, used to store 3D Wii assets as part of the NW4R library."""

    def __init__(self):
        self.files: dict[type[Subfile], list[Subfile]] = {}

    def folder(self, fileType: type[SUBFILE_T]) -> list[SUBFILE_T]:
        """Get the folder of this BRRES for some file type, creating a new one if none exists."""
        try:
            return self.files[fileType]
        except KeyError:
            self.files[fileType] = []
            return self.files[fileType]

    def allFiles(self) -> tuple[Subfile]:
        """One flat tuple containing all the files of this BRRES."""
        return tuple(f for fldr in self.files.values() for f in fldr)

    def search(self, fileType: type[Subfile], fileName: str):
        """Search this BRRES for a subfile with the specified type & name.

        Raise a ValueError if the subfile is not found.
        """
        try:
            return next(f for f in self.files[fileType] if f.name == fileName)
        except (KeyError, StopIteration) as e:
            raise ValueError(f"{fileType} file '{fileName}' not found") from e

    def sort(self):
        """Sort the folders and files of this BRRES filesystem."""
        st = SUBFILE_TYPES
        self.files = {
            t: sorted(self.files[t], key=lambda f: f.name.casefold()) for t in st if t in self.files
        }

    @classmethod
    def unpack(cls, data: bytes) -> "BRRES":
        """Unpack a BRRES file from bytes."""
        return BRRESReader().unpack(data).getInstance()

    def pack(self):
        """Pack this BRRES file to bytes."""
        return BRRESWriter().fromInstance(self).pack()


class BRRESSerializer(Serializer[None, BRRES]):

    MAGIC = b"bres"

    _HEAD_STRCT = Struct(">4s 2s 2x IHH")
    _ROOT_STRCT = Struct(">4s I")

    def __init__(self):
        super().__init__()
        self._stringPool = {}


class BRRESReader(BRRESSerializer, Reader, StrPoolReadMixin):

    _stringPool: dict[int, str]

    _FILE_READERS: dict[str, type[SubfileReader]] = {r.FOLDER_NAME: r for r in (
        MDL0Reader, TEX0Reader, PLT0Reader, CHR0Reader,
        CLR0Reader, PAT0Reader, SRT0Reader, VIS0Reader
    )}

    def __init__(self):
        super().__init__()
        self.files: dict[type[Subfile], dict[str, SubfileReader]] = {}

    def fileName(self, fileReader: SubfileReader):
        """Name of a subfile in this BRRES."""
        return getKey(self.files[fileReader.DATA_TYPE], fileReader)

    def unpack(self, data: bytes):
        super().unpack(data)
        self._data = BRRES()
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, self.offset)
        byteOrder = unpackedHeader[1]
        if byteOrder == b"\xff\xfe":
            raise NotImplementedError("Little-endian BRRES files are currently unsupported")
        rootOffset = unpackedHeader[3] + self.offset
        dictOffset = rootOffset + self._ROOT_STRCT.size
        files = DictReader(self, dictOffset).unpack(data).readEntries(data, DictReader)
        for folderName, folder in files.items():
            try:
                reader = self._FILE_READERS[folderName]
                self.files[reader.DATA_TYPE] = folder.readEntries(data, reader)
            except KeyError:
                print(f"Unsupported BRRES folder '{folderName}' detected & ignored")
        return self

    def _updateInstance(self):
        super()._updateInstance()
        self._data.files = {t: [f.getInstance() for f in d.values()] for t, d in self.files.items()}

    def readString(self, data: bytes, offset: int):
        if offset not in self._stringPool:
            length = int.from_bytes(data[offset - 4 : offset], "big")
            self._stringPool[offset] = data[offset : offset + length].decode("ascii")
        return self._stringPool[offset]


class BRRESWriter(BRRESSerializer, Writer, StrPoolWriteMixin):

    _stringPool: dict[str, int]

    _FILE_WRITERS: dict[type[Subfile], type[SubfileWriter]] = {w.DATA_TYPE: w for w in (
        MDL0Writer, TEX0Writer, PLT0Writer, CHR0Writer,
        CLR0Writer, PAT0Writer, SRT0Writer, VIS0Writer
    )}

    def __init__(self):
        super().__init__()
        self.files: DictWriter[BRRESWriter] = DictWriter(self)

    @classmethod
    def _rootPad(cls, obj):
        """Padding applied to BRRES roots before the file data."""
        return pad(obj, 32, 16)

    @classmethod
    def _finalPad(cls, obj):
        """Padding applied to whole BRRES files once everything's been packed."""
        return pad(obj, 128)

    def stringOffset(self, string: str):
        return self._stringPool.get(string, 0)

    @classmethod
    @cache
    def _packStr(cls, string: str):
        """Pack a string to bytes in the BRRES string format (u32 length followed by string)."""
        return len(string).to_bytes(4, "big") + pad(string.encode("ascii"), 4, extra=True)

    @classmethod
    @cache
    def _strSize(cls, string: str):
        """Get the size in bytes of a string in its BRRES representation."""
        return 4 + pad(len(string), 4, extra=True) # add 4 bc strings are preceded by u32 length

    def fromInstance(self, data: BRRES):
        super().fromInstance(data)
        offset = self.offset
        # generate file/folder writers
        files = {}
        folderOffset = offset + self._HEAD_STRCT.size
        rootFldrSize = DictWriter.sizeFromLen(len(data.files))
        folderSize = sum(DictWriter.sizeFromLen(len(fldr)) for fldr in data.files.values())
        fileOffset = folderOffset + self._rootPad(self._ROOT_STRCT.size + rootFldrSize + folderSize)
        rootFldrOffset = folderOffset + self._ROOT_STRCT.size
        folderOffset = rootFldrOffset + rootFldrSize
        for folderType, folder in data.files.items():
            writer = self._FILE_WRITERS[folderType]
            writers = {} # file writers for this folder
            for file in folder:
                fileWriter = writer(self, fileOffset).fromInstance(file)
                writers[file.name] = fileWriter
                fileOffset += fileWriter.size()
            folderWriter = DictWriter(self, folderOffset).fromInstance(writers)
            files[writer.FOLDER_NAME] = folderWriter
            folderOffset += folderWriter.size()
        self.files = DictWriter(self, rootFldrOffset).fromInstance(files)
        # generate string pool
        offset = fileOffset + 4 # offset of the first string in the pool
        for string in sorted(self.getStrings(), key=lambda s: s.encode("ascii")):
            if string not in self._stringPool:
                self._stringPool[string] = offset
                offset += self._strSize(string)
        self._size = self._finalPad(offset - self.offset)
        return self

    def _calcSize(self):
        return super()._calcSize()

    def getStrings(self):
        return self.files.getStrings()

    def pack(self) -> bytes:
        # pack root w/ main dict holding all resources
        nest = self.files.nest
        packedRes = b"".join(f.pack() for f in nest)
        rootSize = self._ROOT_STRCT.size + len(packedRes)
        packedRoot = self._rootPad(self._ROOT_STRCT.pack(b"root", rootSize) + packedRes)
        # pack files
        allFiles = tuple(f for folder in nest[1:] for f in folder.getInstance().values())
        packedFiles = b"".join(f.pack() for f in allFiles)
        # pack strings
        packedStrs = b"".join(self._packStr(string) for string in self._stringPool)
        # pack header
        rootOffset = self._HEAD_STRCT.size
        bom = b"\xfe\xff"
        numFiles = len(allFiles) + 1 # add 1 bc ig the root is counted as a file
        packedHead = self._HEAD_STRCT.pack(self.MAGIC, bom, self._size, rootOffset, numFiles)
        return super().pack() + self._finalPad(packedHead + packedRoot + packedFiles + packedStrs)
