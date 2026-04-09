from enum import Enum

class AddressSpaceQualifier(Enum):
    STATIC = '__constant'
    DYNAMIC = '__global'

class Buffer:
    def __init__(self, identifier: str, asq: AddressSpaceQualififer = AddressSpaceQualifier.STATIC):
        self.identifier = identifier
        self.asq = asq

        self.size = 0

    def expand(self, amount: int):
        self.size += amount