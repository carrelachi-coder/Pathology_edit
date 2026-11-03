#!/usr/bin/env python3
"""Storage-only review at the exact future output root; no diagnostic launch.

Creates exclusively owned tiny files/empty arm directories, then removes only
those files and empty directories to restore the required fresh-output state.
Never opens the existing reservation for writing or releases its allocation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    a=parser.parse_args();plan=json.loads(a.manifest.read_text())
    spec=importlib.util.spec_from_file_location('_storage_review_controller',plan['controller']['path'])
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    controller.identity_gate(plan);controller.scope_gate(plan)
    root=Path(plan['output_root']);reservation=Path(plan['reservation']['path']);before=reservation.stat()
    root.mkdir(exist_ok=False);root_inode=root.stat().st_ino
    files=[];directories=[];checks={};cleanup=None
    def owned(path,data):
        with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        files.append((path,path.stat().st_ino,hashlib.sha256(data).hexdigest()))
    try:
        owned(root/'.storage_review_owner.json',json.dumps({'purpose':'storage_only','manifest_sha256':controller.sha(a.manifest)}).encode())
        pending=root/'.storage_review.pending';final=root/'.storage_review.final'
        owned(pending,b'preflight only; not an execution marker\n')
        os.link(pending,final);files.append((final,final.stat().st_ino,controller.sha(final)))
        refused=False
        try:os.link(pending,final)
        except FileExistsError:refused=True
        checks['hardlink_no_replace']=refused and pending.stat().st_ino==final.stat().st_ino and controller.sha(pending)==controller.sha(final)
        arm_stats=[]
        for arm in ('A','B','C'):
            directory=root/arm;directory.mkdir(exist_ok=False);directories.append((directory,directory.stat().st_ino))
            marker=directory/'.storage_only.marker';owned(marker,arm.encode())
            st=directory.stat();arm_stats.append({'arm':arm,'resolved_path':str(directory.resolve()),'device':st.st_dev,'inode':st.st_ino,'marker_inode':marker.stat().st_ino})
        checks['arms_distinct']=len({x['resolved_path'] for x in arm_stats})==len({x['inode'] for x in arm_stats})==len({x['marker_inode'] for x in arm_stats})==3
        checks['same_device_as_reservation']=all(x['device']==before.st_dev for x in arm_stats) and root.stat().st_dev==before.st_dev
        checks['live_free_bytes']=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
    finally:
        # No recursive deletion, no deletion of foreign/modified files, no reuse.
        if root.stat().st_ino!=root_inode:raise RuntimeError('root ownership changed')
        for path,inode,expected in reversed(files):
            if path.stat().st_ino!=inode or controller.sha(path)!=expected:
                raise RuntimeError('owned preflight file changed; refuse cleanup')
            path.unlink()
        for path,inode in reversed(directories):
            if path.stat().st_ino!=inode:raise RuntimeError('arm ownership changed')
            path.rmdir()  # Refuses any foreign/nonempty contents.
        root.rmdir();cleanup='owned files and empty directories only; fresh root restored'
    after=reservation.stat()
    checks['reservation_unchanged']=before.st_ino==after.st_ino and before.st_size==after.st_size==plan['reservation']['bytes'] and after.st_blocks*512>=plan['reservation']['bytes']
    controller.identity_gate(plan);controller.scope_gate(plan)
    denied=False
    try:controller.run(plan,controller.sha(a.manifest),json.loads(a.authorization.read_text()))
    except PermissionError:denied=True
    if not denied:raise RuntimeError('authorization unexpectedly opened')
    result={'scope':'exact output-location storage review only; not A/B/C','manifest_sha256':controller.sha(a.manifest),'helper_sha256':controller.sha(__file__),
            'output_root':str(root),'checks':checks,'arm_directory_checks':arm_stats,
            'reservation':{'path':str(reservation),'bytes':after.st_size,'allocated_bytes':after.st_blocks*512,'device':after.st_dev,'inode_before':before.st_ino,'inode_after':after.st_ino,'is_quota':False},
            'cleanup':cleanup,'output_root_exists_after':root.exists(),'identity_scope_recheck_passed':True,
            'authorization_denied_before_launch':denied,'business_modules_loaded':[n for n in sys.modules if n.startswith(('phase3_','inpaint_cells'))],
            'execution_permitted':False,'witness_calls':0}
    controller.write_new(a.report,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
