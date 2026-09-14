#!/usr/bin/env python3
"""Build the development parity harness against SwiftPM's native objects."""
import argparse,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
subprocess.run(['swift','build','--package-path',str(root/'Engine')],check=True)
objects=sorted((root/'Engine/.build/arm64-apple-macosx/debug/H3Native.build').rglob('*.o'))
if not objects:raise RuntimeError('Native build objects missing')
command=['xcrun','clang','-O2','-I'+str(root/'Engine/Vendor/h3.c'),'-I'+str(root/'Engine/Sources/YuE2'),str(root/'Engine/Sources/YuE2/harness/yue2_check.c'),*map(str,objects),'-o',a.output]
for framework in ['Foundation','Metal','MetalPerformanceShaders','MetalPerformanceShadersGraph','Accelerate','AVFoundation','CoreMedia','CoreVideo','VideoToolbox','AudioToolbox','ImageIO','CoreGraphics']:
 command+=['-framework',framework]
command+=['-licucore','-lm'];subprocess.run(command,check=True)
