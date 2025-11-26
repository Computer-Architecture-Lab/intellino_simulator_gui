# path_utils.py
import os

def _find_custom_image_root(start_file: str) -> str:
    """
    start_file 기준으로 상위로 올라가며 'main/custom_image' 폴더가 있는지 탐색.
    발견하면 그 경로를 반환, 없으면 현재 위치에 생성해 반환.
    """
    cur = os.path.abspath(os.path.dirname(start_file))
    for _ in range(12):
        candidate = os.path.join(cur, "main", "custom_image")
        if os.path.isdir(candidate):
            return os.path.normpath(candidate)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # fallback: 현재 파일 기준으로 생성
    fallback = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(start_file)),
                                             "main", "custom_image"))
    os.makedirs(fallback, exist_ok=True)
    return fallback

def get_dirs(start_file: str = __file__):
    """
    Returns:
      custom_image_root, number_image_dir, kmeans_selected_dir
    (없으면 생성)
    """
    root = _find_custom_image_root(start_file)
    number = os.path.join(root, "number_image")
    kmeans = os.path.join(root, "kmeans_selected")
    os.makedirs(number, exist_ok=True)
    os.makedirs(kmeans, exist_ok=True)
    return root, number, kmeans
