import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import glob

def convert_txt_to_npy(txt_path, output_dir):
    """
    하나의 txt 파일을 읽어 (N, 64, 64) 형태의 npy 파일로 변환
    """
    file_name = os.path.basename(txt_path).replace('.txt', '.npy')
    save_path = os.path.join(output_dir, file_name)
    
    data_blocks = []
    current_block = []
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                # 쉼표로 구분된 64개 계수 파싱
                coeffs = [float(x) for x in line.strip().split(',')]
                current_block.append(coeffs)
                
                # 64줄이 쌓이면 하나의 블록으로 확정
                if len(current_block) == 64:
                    data_blocks.append(current_block)
                    current_block = []
        
        if data_blocks:
            final_array = np.array(data_blocks, dtype=np.float32) # 메모리 효율 위해 float32 권장
            np.save(save_path, final_array)
            return f"Success: {file_name} (Shape: {final_array.shape})"
    except Exception as e:
        return f"Error processing {txt_path}: {e}"

def parallel_processing(input_dir, output_dir, max_workers=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"Total files to process: {len(txt_files)}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 각 파일에 대해 병렬 작업 수행
        results = list(executor.map(convert_txt_to_npy, txt_files, [output_dir]*len(txt_files)))
    
    for res in results:
        print(res)

if __name__ == "__main__":
    # 설정값
    INPUT_FOLDER = "path/to/your/txt_data"
    OUTPUT_FOLDER = "path/to/your/npy_data"
    
    parallel_processing(INPUT_FOLDER, OUTPUT_FOLDER, max_workers=os.cpu_count())



















import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

class DCTDataset(Dataset):
    def __init__(self, npy_dir):
        self.file_list = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        self.data_info = [] # (file_index, block_index) 저장
        
        # 파일별로 몇 개의 블록이 있는지 미리 인덱싱
        for f_idx, f_path in enumerate(self.file_list):
            # mmap으로 열면 파일 전체를 읽지 않고 헤더만 확인하여 매우 빠름
            data = np.load(f_path, mmap_mode='r')
            num_blocks = data.shape[0]
            for b_idx in range(num_blocks):
                self.data_info.append((f_idx, b_idx))
        
        # 실제 데이터는 호출될 때 mmap으로 접근하기 위해 핸들 관리
        self.mmaps = [np.load(f, mmap_mode='r') for f in self.file_list]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        f_idx, b_idx = self.data_info[idx]
        # 필요한 블록만 메모리에 매핑하여 가져옴 (매우 빠름)
        sample = self.mmaps[f_idx][b_idx]
        
        return torch.from_numpy(sample.copy()).float()

# 사용 예시
# dataset = DCTDataset("./npy_data")
# loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)


