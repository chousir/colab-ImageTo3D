"""Batch image-to-3D pipeline for free Colab T4 (MV-Adapter views, Hunyuan3D-2mv shape and paint).

Each stage runs in its own Python process so model memory is fully returned to the OS between stages.
"""
