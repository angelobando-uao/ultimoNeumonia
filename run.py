#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run.py - Punto de entrada universal para ejecutar UAO-Neumonia."""

import sys
import os

# Asegurar que el directorio del proyecto esté en sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_neumonia import main

if __name__ == "__main__":
    sys.exit(main())
