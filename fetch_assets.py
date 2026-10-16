"""Download only the upstream models used by simulation.py."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET


ASSET_DIR = Path(__file__).resolve().parent / 'assets' / 'vendor'
SOURCES = {
    'panda': ('google-deepmind/mujoco_menagerie',
              '62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35',
              'franka_emika_panda'),
    'drawer': ('robocasa/robocasa',
               'a9e152e2eea16d342dac40e4577fb4c68b932854',
               'robocasa/models/assets/fixtures/cabinets'),
    'cup': ('Lifelong-Robot-Learning/LIBERO',
            '8f1084e3132a39270c3a13ebe37270a43ece2a01',
            'notebooks/custom_assets/libero_mug'),
}


def download(source:str, filename:str, force:bool=False,
             repository_root:bool=False) -> Path:
    """Fetch one pinned upstream file and return its local path."""
    repository, commit, directory = SOURCES[source]
    remote_path = filename if repository_root else f'{directory}/{filename}'
    target = ASSET_DIR / source / filename
    if target.is_file() and target.stat().st_size and not force:
        return target

    url = f'https://raw.githubusercontent.com/{repository}/{commit}/{remote_path}'
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'cup-drawer-demo'})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    temporary = target.with_suffix(target.suffix + '.part')
    temporary.write_bytes(content)
    temporary.replace(target)
    print(f'Downloaded {target.relative_to(ASSET_DIR)}', flush=True)
    return target


def fetch_assets(force:bool=False) -> None:
    """Download the robot, drawer template, mug, and their license files."""
    panda = download('panda', 'panda.xml', force)
    meshes = {mesh.attrib['file'] for mesh in ET.parse(panda).iter('mesh')}
    jobs = [('panda', f'assets/{mesh}', False) for mesh in sorted(meshes)]
    jobs += [('panda', name, False) for name in ['LICENSE', 'README.md']]
    jobs += [('drawer', 'drawer.xml', False), ('drawer', 'LICENSE', True)]
    jobs += [('cup', name, False) for name in
             ['libero_mug.xml', 'libero_mug.obj', 'libero_mug.png']]
    jobs += [('cup', 'LICENSE', True)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, source, name, force, root)
                   for source, name, root in jobs]
        for future in futures:
            future.result()
    print(f'Assets ready: {ASSET_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Replace cached files.')
    args = parser.parse_args()
    fetch_assets(args.force)
