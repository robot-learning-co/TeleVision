#! /usr/bin/bash
{
    cloudflared tunnel --protocol http2 run television
}&
eval "$(conda shell.bash hook)"
conda activate trlc3.10
python /home/jannik/repos/TeleVision/teleop/teleop_active_cam.py