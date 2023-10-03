# standard imports
from struct import Struct
from typing import TypeVar
# internal imports
from .binaryutils import strToInt, maxDifBit, calcOffset, maxBitVal
from .serialization import (
    S_PARENT_T, Readable, Writable, Serializer, Reader, Writer, StrPoolReadMixin, StrPoolWriteMixin
)


_R_STR_T = TypeVar("_R_STR_T", "Readable", "StrPoolReadMixin")
_W_STR_T = TypeVar("_W_STR_T", "Writable", "StrPoolWriteMixin")
_S_ENTRY_T = TypeVar("_S_ENTRY_T")
_R_ENTRY_T = TypeVar("_R_ENTRY_T", bound="Readable")


class DictSerializer(Serializer[S_PARENT_T, dict[str, _S_ENTRY_T]]):
    """Serializer for an ordered map with string keys stored in a BRRES file."""
    DATA_TYPE = dict
    _HEAD_STRCT = Struct(">II")


class DictReader(DictSerializer[_R_STR_T, int], Reader, StrPoolReadMixin):
    """Reader for an ordered map with string keys stored in a BRRES file."""

    def __init__(self, parent: S_PARENT_T = None, offset = 0):
        super().__init__(parent, offset)
        self._duplicateName = ""
        """If any entry names are duplicated, one is stored here to be used for an error message."""

    def unpack(self, data: bytes):
        super().unpack(data)
        offset = self._offset
        self._data = {}
        self._duplicateName = ""
        unpackedHeader = self._HEAD_STRCT.unpack_from(data, offset)
        numEntries = unpackedHeader[1]
        entrySize = EntryStruct.size()
        firstEntryOffset = offset + self._HEAD_STRCT.size + entrySize
        lastEntryOffset = firstEntryOffset + entrySize * numEntries
        # parse entry names & add offsets to dict, keeping track of duplicate names if any
        for entryOffset in range(firstEntryOffset, lastEntryOffset, entrySize):
            entry = EntryStruct().unpack(data[entryOffset:])
            nameOffset = offset + entry.nameOffset
            dataOffset = offset + entry.dataOffset
            name = self.readString(data, nameOffset)
            if name in self._data:
                self._duplicateName = name
            self._data[name] = dataOffset
        return self

    def readEntries(self, data: bytes, entryType: type[_R_ENTRY_T]):
        """Unpack this dict's entries and return a dict containing them.

        The entry type is required to determine how to unpack the entries. Their parents will be
        set to the parent of this dict. If any of this dict's entries point to the same offset,
        they'll be given the same reader object. 
        """
        if self._duplicateName:
            raise ValueError(f"Cannot unpack BRRES collection of type '{entryType.__name__}' "
                             f"containing multiple entries with the same name "
                             f"('{self._duplicateName}')")
        unpackedByOffset: dict[int, _R_ENTRY_T] = {}
        unpackedByKey: dict[str, _R_ENTRY_T] = {}
        for key, offset in self._data.items():
            if offset not in unpackedByOffset:
                unpackedByOffset[offset] = entryType(self.parentSer, offset).unpack(data)
            unpackedByKey[key] = unpackedByOffset[offset]
        return unpackedByKey


class DictWriter(DictSerializer[_W_STR_T, Writer], Writer, StrPoolWriteMixin):
    """Writer for an ordered map with string keys stored in a BRRES file."""

    @classmethod
    def sizeFromLen(cls, length: int):
        """Calculate the size in bytes of a dict with [length] entries."""
        return cls._HEAD_STRCT.size + EntryStruct.size() * (length + 1)

    def _calcSize(self):
        return self.sizeFromLen(len(self._data))

    def getStrings(self):
        entries = self._data.values()
        strs = set(self._data.keys())
        return strs.union(*(e.getStrings() for e in entries if isinstance(e, StrPoolWriteMixin)))

    @property
    def nest(self):
        """Tuple containing this writer as well as any nested dict writer entries."""
        nest = tuple(ne for e in self._data.values() if isinstance(e, DictWriter) for ne in e.nest)
        return (self, ) + nest

    def pack(self):
        # 1 entry is added to the beginning to act as a root for the binary search tree
        processedEntries = [EntryStruct()]
        for entryName, entryData in self._data.items():
            processedEntry = EntryStruct.generate(self._data, processedEntries)
            processedEntry.dataOffset = calcOffset(self.offset, entryData.offset)
            processedEntry.nameOffset = calcOffset(self.offset, self.stringOffset(entryName))
            processedEntries.append(processedEntry)
        packedEntries = b"".join(entry.pack() for entry in processedEntries)
        packedHeader = self._HEAD_STRCT.pack(self._size, len(processedEntries) - 1)
        return super().pack() + packedHeader + packedEntries


class EntryStruct(Readable, Writable):
    """Entry in a BRRES dict.

    This is structured the way that BRRES dict entries are structured in packed form, and it is only
    to be used in BRRES dicts internally for unpacking/packing.
    """

    _STRCT = Struct(">HxxHHII")

    def __init__(self, idx = 0):
        super().__init__()
        self.id = maxBitVal(16) # id for comparisons in binary tree traversal (defaults to max)
        self.idxL = idx # index of left child in parent dict (if no children, is this entry's idx)
        self.idxR = idx # index of right child in parent dict (if no children, is this entry's idx)
        self.nameOffset = 0 # offset to entry's name in brres, relative to start of parent dict
        self.dataOffset = 0 # offset to entry's data in brres, relative to start of parent dict

    def unpack(self, data: bytes):
        unpacked = self._STRCT.unpack_from(data)
        self.id, self.idxL, self.idxR, self.nameOffset, self.dataOffset = unpacked
        return self

    @classmethod
    def size(cls): # classmethod override is fine imo - pylint: disable=arguments-differ
        return cls._STRCT.size

    def pack(self):
        return self._STRCT.pack(self.id, self.idxL, self.idxR, self.nameOffset, self.dataOffset)

    def getIDBit(self, entryName: str):
        """For some name, get its bit to which this entry's ID points."""
        charIdx = self.id >> 3
        bitIdx = self.id & 0b111
        return charIdx < len(entryName) and strToInt(entryName[charIdx]) >> bitIdx & 1

    @classmethod
    def calcID(cls, n1: str, n2: str):
        """Calculate an entry ID for the search tree based on a comparison of two entries' names.

        The entry corresponding to the first name provided should receive the ID.

        This 16-bit ID is made up of two components:
        Most significant 13 bits: The index of the last character that differs between the two names
        (or the last possible index in n1, if it's longer than n2)
        Least significant 3 bits: The index of the most significant bit that differs between the
        characters at that index in each name
        """
        charIdx = len(n1) - 1
        if len(n1) <= len(n2):
            charIdx = [i for i, (c1, c2) in enumerate(zip(n1, n2)) if c1 != c2][-1]
        bitIdx = maxDifBit(strToInt(n1[charIdx]), strToInt(n2[charIdx]) if charIdx < len(n2) else 0)
        return (charIdx << 3) | (bitIdx)

    @classmethod
    def generate(cls, d: dict[str], entries: list["EntryStruct"]):
        """Generate a struct for a BRRES dict entry and calculate its ID & left/right indices.

        A dict and a list of other structs (each corresponding to one of the dict's entries up to
        this one) are required.

        Structs in the aforementioned list may have their left & right indices changed as this
        entry is inserted into the binary search tree.

        Further detail: BRRES dict entries are stored in a binary search tree, described by the id
        and left & right indices of each entry. This method exists for that tree's creation. To
        be honest, I don't entirely understand how it works, so a lot of the black magic here is
        based on the sources of BrawlBox and SZSTools. I need to revisit this in the future.
        """
        # list of entry names
        names = [""] + list(d.keys())
        # entry: entry being inserted into tree (ultimately returned)
        # cur: current entry in tree traversal, being compared to entry
        # prev: previous cur before this one
        entryIdx = len(entries)
        entry = cls(entryIdx)
        prevIdx = 0
        curIdx = entries[prevIdx].idxL
        entry.id = cls.calcID(names[entryIdx], names[prevIdx])
        # previous direction travelled in traversal
        isRight = False
        # go until entry is right of current or current is right of previous
        while entry.id <= entries[curIdx].id and entries[curIdx].id < entries[prevIdx].id:
            # if entry and current have the same id, calculate a new one for entry
            if entry.id == entries[curIdx].id:
                entry.id = cls.calcID(names[entryIdx], names[curIdx])
                if entry.getIDBit(names[curIdx]):
                    entry.idxL = entryIdx
                    entry.idxR = curIdx
                else:
                    entry.idxL = curIdx
                    entry.idxR = entryIdx
            # go to the next node
            prevIdx = curIdx
            isRight = entries[curIdx].getIDBit(names[entryIdx])
            if isRight: # is new cur node right of prev node?
                curIdx = entries[curIdx].idxR
            else:
                curIdx = entries[curIdx].idxL
        # done with tree traversal, now update indices in entry and its parent
        if len(names[curIdx]) == len(names[entryIdx]) and entry.getIDBit(names[curIdx]):
            entry.idxR = curIdx
        else:
            entry.idxL = curIdx
        if isRight:
            entries[prevIdx].idxR = entryIdx
        else:
            entries[prevIdx].idxL = entryIdx
        return entry
