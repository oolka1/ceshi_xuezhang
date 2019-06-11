#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:28:50 2019

@author: hesun
"""

import sys
import os.path as osp


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
add_path(osp.join(this_dir, '..'))