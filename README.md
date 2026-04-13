<div align="center">
  <h1>🎮 Omniteleop - Teleoperation Stack for Dexmate Robots</h1>
</div>

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

## 📦 Installation

```shell
cd omniteleop/
pip install -e .
```

## ✨ Features

- 🕹️ **JoyCon Controller Support** - Use Nintendo JoyCon for robot control
- 💪 **Exoskeleton Arm Control** - Intuitive arm teleoperation via Dynamixel exoskeleton
- 🛡️ **Safety System** - Built-in emergency stop and joint limits enforcement
- 📹 **Data Collection** - Record teleoperation data for policy learning
- 🔄 **Trajectory Replay** - Replay recorded robot trajectories
- 📊 **Telemetry Viewer** - Real-time visualization of joint data
- 😎 **GUI** - Easy to use WebApp that provides an intuitive, visual layer to the entire repository.

## 🚀 Quick Start

```shell
omni-arm       # Exoskeleton arm reader
omni-joycon    # JoyCon controller reader
omni-cmd       # Command processor with safety
omni-robot     # Robot controller
omni-recorder  # MDP recorder for policy learning
omni-telemetry # Telemetry viewer

# alpha
app/launch.sh  # One command that replaces all
```

## 📄 Licensing

This project is **dual-licensed**:

### 🔓 Open Source License
This software is available under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
See the [LICENSE](./LICENSE) file for details.

### 💼 Commercial License
For businesses that want to use this software in proprietary applications without the AGPL requirements, commercial licenses are available.

**📧 Contact us for commercial licensing:** contact@dexmate.ai

---

<div align="center">
  <h3>🤝 Ready to teleoperate robots?</h3>
  <p>
    <a href="mailto:contact@dexmate.ai">📧 Contact Us</a>
  </p>
</div>

