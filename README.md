# OpenArm Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

See [openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab) for original repo.

We updated the visualization of bi-arm to achieve realtime smooth tracking of eef trajectories.

```bash
python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Reach-OpenArm-Bi-Play-v0  --num_envs 10 --checkpoint openarm_bi_reach.pth
```
See gif

![](openarm_ik.gif)

## License

[Apache License 2.0](LICENSE.txt)

Copyright 2025 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
