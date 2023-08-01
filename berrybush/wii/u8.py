# standard imports
from struct import Struct
from typing import Union
# internal imports
from .binaryutils import pad


class InvalidU8IndexError(TypeError):
    """Invalid type for a U8 directory key/index"""

    def __init__(self, key):
        super().__init__(f"Type '{type(key)}' is invalid as a U8 directory key/index")


class U8Directory(dict[str, Union["U8Directory", bytes]]):
    """Alphabetically sorted dict that supports numerical indexing. Represents a filesystem.

    Keys (folder/file names) should be strings. Values (folder/file data) should be U8 dirs for
    folders and bytes-like objects for files, though value type is not checked.
    """

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(list(self.keys())[key])
        if isinstance(key, str):
            return super().__getitem__(key)
        raise InvalidU8IndexError(key)

    def __delitem__(self, key):
        if isinstance(key, int):
            super().__delitem__(list(self.keys())[key])
        elif isinstance(key, str):
            super().__delitem__(key)
        else:
            raise InvalidU8IndexError(key)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            super().__setitem__(list(self.keys())[key], value)
        elif isinstance(key, str):
            super().__setitem__(key, value)
        else:
            raise InvalidU8IndexError(key)

    def hasSubDir(self, subDir: "U8Directory"):
        """Return true if the given subdirectory exists anywhere within this one."""
        if subDir.items() <= self.items():
            return True
        for node in self.values():
            if isinstance(node, U8Directory) and node.hasSubDir(subDir):
                return True
        return False

    def getNodeList(self) -> list["U8Directory"]:
        """Return a list of all folders and files contained within this directory.

        Note that folders in this list still have their items, so nested files/folders are bound to
        kind-of show up multiple times (within the folders and then once on their own in the list).
        """
        output = []
        for name, node in self.items():
            output.append(U8Directory({name: node}))
            if isinstance(node, U8Directory):
                output += node.getNodeList()
        return output

    def __str__(self):
        # similar to normal dict __str__ method, but file data is truncated for readability
        items = []
        for name, node in self.items():
            if isinstance(node, U8Directory): # folders
                items.append(f"'{name}': {node}")
            else: # files (should always be bytes-like)
                items.append(f"'{name}': 0x{node[:4].hex()}...")
        return "{" + ", ".join(items) + "}"


class U8():
    """Wii archive file made up of folders & files."""

    MAGIC = b"U\xaa8-"

    _HEAD_STRCT = Struct(">4s III 16x")
    _NODE_STRCT = Struct(">?xHII")

    def __init__(self, filesystem: U8Directory = None):
        super().__init__()
        self.filesystem = filesystem if filesystem is not None else U8Directory({})

    @classmethod
    def unpack(cls, data: bytes, offset = 0):
        """Unpack a U8 file from bytes."""
        unpackedHeader = cls._HEAD_STRCT.unpack_from(data, offset)
        rootOffset = unpackedHeader[1] + offset
        unpackedRoot = cls._NODE_STRCT.unpack_from(data, rootOffset)
        numNodes = unpackedRoot[3]
        nameListOffset = rootOffset + cls._NODE_STRCT.size * numNodes
        # generate node list
        u8 = cls()
        u8.filesystem = cls._unpackDirectory(data, rootOffset, rootOffset, nameListOffset)[""]
        return u8

    @classmethod
    def _unpackDirectory(cls, data: bytes, nodeOffset: int, rootOffset: int, nameListOffset: int):
        """Unpack a directory from bytes, starting at a specific node.

        Takes U8 data (the entire file), the node's offset, the offset of the first node ("root"),
        and the offset of the node name list.
        """
        unpackedNode = cls._NODE_STRCT.unpack_from(data, nodeOffset)
        # read null-terminated name
        nameOffset = unpackedNode[1]
        nameData = data[nameListOffset + nameOffset:]
        nodeName = nameData[:nameData.index(b"\x00")].decode("ascii")
        # if not folder, return dir w/ just this file
        isFolder = unpackedNode[0]
        if not isFolder:
            dataOffset, dataSize = unpackedNode[2:4]
            nodeData = data[dataOffset : dataOffset + dataSize]
            return U8Directory({nodeName: nodeData})
        # if folder, return dir w/ folder name leading to another dir w/ all sub-files/folders
        dirEndIdx = unpackedNode[3]
        dirEndOffset = rootOffset + dirEndIdx * cls._NODE_STRCT.size
        folder = U8Directory({nodeName: U8Directory({})})
        subNodeOffset = nodeOffset + cls._NODE_STRCT.size
        while subNodeOffset < dirEndOffset: # unpack sub-nodes
            subdir = cls._unpackDirectory(data, subNodeOffset, rootOffset, nameListOffset)
            folder[nodeName] |= subdir
            subNodeOffset += cls._NODE_STRCT.size
            # if the sub-node is a folder, skip over its sub-nodes so they're not duplicated
            if isinstance(subdir[0], U8Directory):
                subNodeOffset += cls._NODE_STRCT.size * len(subdir[0])
        return folder

    def pack(self):
        """Pack this U8 file to bytes."""
        nodes = U8Directory({"": self.filesystem}).getNodeList()
        nodeList = b""
        nameList = b""
        dataList = b""
        nameListSize = sum(len(next(iter(node.keys()))) + 1 for node in nodes)
        numNodes = len(nodes)
        nodeInfoLen = numNodes * self._NODE_STRCT.size + nameListSize
        dataOffset = self._HEAD_STRCT.size + pad(nodeInfoLen, 16)
        for nodeIdx, node in enumerate(nodes):
            name, data = next(iter(node.items()))
            isFolder = isinstance(data, U8Directory)
            nameOffset = len(nameList)
            if isFolder:
                # folders require index of parent folder and index of next node not in this folder
                nx = nodeIdx + 1 # next node after this one
                try:
                    parent = [i for i, n in enumerate(nodes[:nodeIdx]) if n.hasSubDir(node)][-1]
                except IndexError:
                    parent = 0
                try:
                    dirEnd = [nx + i for i, n in enumerate(nodes[nx:]) if not node.hasSubDir(n)][0]
                except IndexError:
                    dirEnd = len(nodes)
                nodeList += self._NODE_STRCT.pack(isFolder, nameOffset, parent, dirEnd)
            else:
                # files are simpler to pack, just needing data offset & size
                # data is padded to 32, unless it's the last entry
                nodeDataOffset = dataOffset + len(dataList)
                nodeList += self._NODE_STRCT.pack(isFolder, nameOffset, nodeDataOffset, len(data))
                dataList += pad(data, 32) if nodeIdx != numNodes - 1 else data
            nameList += name.encode("ascii") + b"\x00"
        header = self._HEAD_STRCT.pack(self.MAGIC, self._HEAD_STRCT.size, nodeInfoLen, dataOffset)
        return header + pad(nodeList + nameList, 16) + dataList
