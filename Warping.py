import torch
import torch.nn.functional as F

def warp_high_precision(ref_16x, flow, scale_factor=16.0):
    """
    Args:
        ref_16x: (B, C, 16H, 16W) - 16배 확대된 Reference Frame
        flow: (B, 2, H, W) - (dx, dy) 순서의 Optical Flow (Pixel unit @ HxW scale)
        scale_factor: 16.0 - Reference Frame의 확대 비율

    Returns:
        warped_image: (B, C, H, W) - Warping된 결과
    """
    B, C, H_ref, W_ref = ref_16x.shape
    B_flow, _, H, W = flow.shape

    # 1. 기본 메쉬 그리드 생성 (H x W 해상도)
    # range: 0 ~ W-1, 0 ~ H-1
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=ref_16x.device, dtype=ref_16x.dtype),
        torch.arange(W, device=ref_16x.device, dtype=ref_16x.dtype),
        indexing='ij'
    )
    
    # (B, 2, H, W) 형태로 맞춤 (x, y 순서)
    base_grid = torch.stack([x_grid, y_grid], dim=0) # (2, H, W)
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1) # (B, 2, H, W)

    # 2. Flow 적용 (Warping 좌표 계산 @ Low-Res Scale)
    # flow 값은 pixel 단위라고 가정합니다. 
    # 만약 flow가 integer-coded된 1/16 단위 정수라면 flow / 16.0을 먼저 해주세요.
    warped_grid = base_grid + flow 

    # 3. High-Res 좌표계로 스케일링
    # Reference 이미지가 16배 크므로, 좌표도 16배 해줍니다.
    warped_grid_high = warped_grid * scale_factor

    # 4. Grid Sample을 위한 정규화 ([-1, 1] 범위)
    # 정규화 기준은 Reference Frame의 width, height를 사용해야 합니다.
    # (B, 2, H, W) -> (B, H, W, 2) 순서 변경 (grid_sample 요구사항)
    vgrid = warped_grid_high.permute(0, 2, 3, 1)
    
    # 공식: 2 * (coords / (size - 1)) - 1
    # x 좌표 정규화
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / (W_ref - 1) - 1.0
    # y 좌표 정규화
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / (H_ref - 1) - 1.0

    # 5. Sampling
    # align_corners=True: -1은 픽셀 0, +1은 픽셀 -1에 매핑
    # mode='bilinear': 16배 커진 이미지에서 소수점 좌표를 가져오므로 bilinear가 적합
    output = F.grid_sample(ref_16x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)

    return output

# --- 사용 예시 ---
# 가정:
# Batch=1, Channel=3, Height=64, Width=64
# Ref Frame은 1024x1024 (16배)
# Flow는 64x64 크기

B, C, H, W = 1, 3, 64, 64
scale = 16

ref_frame_high = torch.randn(B, C, H * scale, W * scale).cuda() # GPU 사용 시
flow_data = torch.randn(B, 2, H, W).cuda() # 임의의 Flow

# 실행
warped_output = warp_high_precision(ref_frame_high, flow_data, scale_factor=16)

print(f"Input Ref Shape: {ref_frame_high.shape}")
print(f"Output Shape: {warped_output.shape}") # (1, 3, 64, 64)가 나와야 함
