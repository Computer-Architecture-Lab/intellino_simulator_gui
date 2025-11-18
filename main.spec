# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('C:\\Users\\alsxo\\miniconda3\\envs\\intellino\\Lib\\site-packages\\PySide2\\plugins\\imageformats', 'PySide2\\plugins\\imageformats'), ('main\\intellino_TM.png', 'main'), ('main\\intellino_TM_transparent.png', 'main'), ('main\\home.png', 'main'), ('main\\custom_image', 'main\\custom_image')]
datas += collect_data_files('matplotlib')


block_cipher = None


a = Analysis(['main\\main.py'],
             pathex=[],
             binaries=[],
             datas=datas,
             hiddenimports=['mnist'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['torch', 'torchvision', 'torchaudio', 'torch._C'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
