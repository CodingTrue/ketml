import pyopencl as cl
from enum import Enum

from ketml.utils import get_cl_info

def get_cl_device_name(cl_device: cl.Device) -> str:
    return (
        get_cl_info(target=cl_device, key=cl.device_info.BOARD_NAME_AMD) or cl_device.get_info(cl.device_info.NAME)
    ).strip()

class DeviceType(Enum):
    CPU = cl.device_type.CPU
    GPU = cl.device_type.GPU
    ALL = -1

class Device:
    _default: Device = None

    def __init__(self, cl_device: cl.Device):
        self.cl_device = cl_device

        self.device_name = get_cl_device_name(cl_device=cl_device)

    def use(self):
        Device._default = self

    def get_name(self) -> str:
        return self.device_name

    def __repr__(self):
        return f"Device('{self.device_name}')"

    @classmethod
    def get_default(cls) -> Device:
        if cls._default is None:
            raise Exception("Default device is not set.")

        return cls._default

    @staticmethod
    def devices(device_type: DeviceType) -> list[Device]:
        target_types = [device_type] if device_type is not DeviceType.ALL else [DeviceType.CPU, DeviceType.GPU]
        hits = []

        for target_type in target_types:
            for platform in cl.get_platforms():
                for device in platform.get_devices(device_type=target_type.value):
                    hits.append(Device(cl_device=device))
        return hits