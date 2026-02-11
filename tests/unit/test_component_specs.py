from contracts.codesign.component import ActuatorComponentSpec, SensorComponentSpec
from contracts.codesign.component import PortDirection, DataType


def test_rgb_camera_component_ports_and_dimensions():
    cam = SensorComponentSpec.rgb_camera(
        name="cam",
        resolution=(640, 480),
        fps=30.0,
        fov_degrees=90.0,
        latency_ms=10.0,
    )

    assert cam.interface.name == "cam_interface"
    assert len(cam.interface.ports) == 2

    rgb = next(p for p in cam.interface.ports if p.name == "rgb_image")
    assert rgb.direction == PortDirection.OUTPUT
    assert rgb.data_type == DataType.IMAGE
    assert rgb.dimension == (640, 480, 3)


def test_wheeled_base_component_has_cmd_vel_port():
    base = ActuatorComponentSpec.wheeled_base(
        name="base",
        max_linear_velocity_mps=1.0,
        max_angular_velocity_rps=2.0,
        control_rate_hz=20.0,
    )
    assert any(p.name == "cmd_vel" for p in base.interface.ports)

