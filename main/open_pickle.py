import pickle
from intellino.core.neuron_cell import NeuronCells
import pandas as pd

# Load the pickle file
with open("trained_neuron.pkl", "rb") as f:
    neuron_cells = pickle.load(f)

# 2. 전체 속성 딕셔너리 추출
attributes = neuron_cells.__dict__

# 3. 보기 좋게 속성 이름, 타입, 요약 내용 출력
print("📦 NeuronCells 객체의 속성 정보")
for key, value in attributes.items():
    preview = str(value)
    if len(preview) > 100:
        preview = preview[:100] + " ..."
    print(f"{key:<30} | {type(value).__name__:<20} | {preview}\n")

# 첫 번째 셀의 속성 보기
first_cell = neuron_cells.cells[0]

print("🧠 첫 번째 Cell 객체의 속성 정보")
for key, value in first_cell.__dict__.items():
    preview = str(value)
    if len(preview) > 100:
        preview = preview[:100] + " ..."
    print(f"{key:<25} | {type(value).__name__:<20} | {preview}")

print("\n")

# 전체 Cell 정보 수집
all_cells_summary = []

for idx, cell in enumerate(neuron_cells.cells):
    cell_info = {"index": idx}
    for key, value in cell.__dict__.items():
        preview = str(value)
        if len(preview) > 100:
            preview = preview[:100] + " ..."
        cell_info[key] = preview
    all_cells_summary.append(cell_info)

# 데이터프레임으로 보기 좋게 정리
df = pd.DataFrame(all_cells_summary)
print(df.head())       # 처음 몇 개 출력
df.to_csv("all_neuron_cells.csv", index=False)  # 전체 저장도 가능


neuron_cells.inference

