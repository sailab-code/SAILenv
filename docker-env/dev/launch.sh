#!/bin/bash

cd lve && xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' ./lve.x86_64