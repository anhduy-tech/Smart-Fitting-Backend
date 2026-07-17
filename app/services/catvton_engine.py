"""
catvton_engine.py
------------------
Gần như NGUYÊN VĂN file tryon_service.py đã test chạy thành công (40/40 bước,
verify bằng traceback thật trên máy dev) khi chạy độc lập trong ~/catvton.
CHỈ khác 2 chỗ so với bản gốc:
  1. Thêm _ensure_catvton_on_path() để tìm được `model`/`utils` khi chạy từ
     Django (không còn đứng ở gốc repo ~/catvton nữa).
  2. Thêm skip_safety_check=True (project đã có moderation.py riêng, tránh
     lỗi thiếu file NSFW.jpg placeholder + tiết kiệm VRAM/RAM).

KHÔNG sửa gì thêm logic bên trong TryOnEngine — giữ nguyên bản đã verify.

Dùng cho cả 2 trường hợp:
  - upper (áo)
  - lower (quần)
  - overall (bộ đồ / váy liền)

Độ chính xác cao phụ thuộc nhiều vào:
  1. Ảnh người: đứng thẳng, nền đơn giản, ánh sáng đều, full-body hoặc half-body
     tuỳ loại quần áo.
  2. Ảnh quần áo: chụp phẳng (flat-lay) hoặc trên mannequin, nền trắng/đơn sắc
     cho kết quả tốt nhất.
  3. Mask: AutoMasker tự sinh mask qua DensePose + SCHP, nhưng nếu ảnh người
     phức tạp (nhiều lớp áo, phụ kiện che), nên cho phép người dùng vẽ tay mask
     (xem app.py gốc — ImageEditor) để tăng độ chính xác.
"""

import os
import sys
import io
from typing import Literal, Optional

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers.image_processor import VaeImageProcessor

# Them thu muc nay vao sys.path de import duoc `model.xxx` / `utils` dung
# nguyen ban goc repo CatVTON (model/pipeline.py, model/cloth_masker.py...
# nam ngay ben canh file nay, trong app/services/catvton_lib/).
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catvton_lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from model.pipeline import CatVTONPipeline, CatVTONPix2PixPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

ClothType = Literal["upper", "lower", "overall"]




class TryOnEngine:
    """
    Khởi tạo 1 lần (nặng, load model lên GPU), gọi run() nhiều lần cho từng request.

    Cấu hình mặc định TỐI ƯU CHO GPU 6GB VRAM (RTX 4050 laptop):
      - Dùng checkpoint CatVTON-MaskFree -> KHÔNG cần load DensePose/SCHP
        (tiết kiệm cả VRAM lẫn RAM, không cần cài detectron2).
      - fp16 thay vì bf16 (nhẹ hơn 1 chút, Ada Lovelace hỗ trợ cả 2).
      - Resolution mặc định giảm còn 576x768 thay vì 768x1024 gốc.
      - Bật attention/vae slicing để giảm đỉnh VRAM khi decode.

    Khi mask_free=False, cần person_mask được truyền vào thủ công (PIL "L" mode,
    trắng = vùng cần thay đồ) vì không có AutoMasker để tự sinh mask nữa.
    Bạn có thể vẽ tay mask này bằng công cụ đơn giản (vd: xoá nền + threshold)
    hoặc để user tự vẽ trên UI (giống ImageEditor trong app.py gốc).
    """

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        mask_free: bool = True,
        catvton_repo: Optional[str] = None,
        mixed_precision: str = "fp16",  # "no" | "fp16" | "bf16"
        device: str = "cuda",
        use_tf32: bool = True,
        low_vram: bool = True,
    ):
        self.device = device
        self.mask_free = mask_free

        # QUAN TRỌNG: 2 pipeline dùng 2 base UNet khác nhau về kiến trúc (số kênh input):
        #   - CatVTONPipeline (có mask)   -> UNet inpainting 9 kênh -> base "booksforcharlie/stable-diffusion-inpainting"
        #   - CatVTONPix2PixPipeline (mask-free) -> UNet pix2pix 8 kênh -> base "timbrooks/instruct-pix2pix"
        # Dùng sai base sẽ lỗi conv_in channel mismatch (đã gặp khi test).
        if base_model_path is None:
            base_model_path = (
                "timbrooks/instruct-pix2pix" if mask_free
                else "booksforcharlie/stable-diffusion-inpainting"
            )

        if catvton_repo is None:
            catvton_repo = "zhengchong/CatVTON-MaskFree" if mask_free else "zhengchong/CatVTON"

        repo_path = snapshot_download(repo_id=catvton_repo)

        # LƯU Ý: CatVTONPix2PixPipeline (mask-free) có bug nhỏ trong auto_attn_ckpt_load —
        # nó dùng version làm TÊN THƯ MỤC TRỰC TIẾP thay vì map qua tên đầy đủ như
        # CatVTONPipeline gốc. Repo HuggingFace vẫn đặt tên thư mục đầy đủ
        # (mix-48k-1024 / vitonhd-16k-512 / dresscode-16k-512) nên phải truyền
        # đúng tên thư mục đó, không phải alias ngắn "mix"/"vitonhd"/"dresscode".
        attn_version_map = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }
        attn_ckpt_version = attn_version_map["mix"] if mask_free else "mix"

        pipeline_cls = CatVTONPix2PixPipeline if mask_free else CatVTONPipeline
        self.pipeline = pipeline_cls(
            base_ckpt=base_model_path,
            attn_ckpt=repo_path,
            attn_ckpt_version=attn_ckpt_version,
            weight_dtype=init_weight_dtype(mixed_precision),
            use_tf32=use_tf32,
            device=device,
            skip_safety_check=True,  # da co moderation.py rieng trong project
        )

        # Giảm đỉnh VRAM khi decode ảnh độ phân giải cao — rất quan trọng với 6GB VRAM
        if low_vram and hasattr(self.pipeline, "unet"):
            try:
                self.pipeline.unet.enable_attention_slicing("max")
            except Exception:
                pass
        if low_vram and hasattr(self.pipeline, "vae"):
            try:
                self.pipeline.vae.enable_slicing()
                self.pipeline.vae.enable_tiling()
            except Exception:
                pass

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

        self.automasker = None
        if not mask_free:
            # Chỉ load AutoMasker (DensePose+SCHP, tốn thêm ~1-2GB VRAM) nếu
            # thật sự cần auto-mask và máy đủ VRAM (>=8GB khuyến nghị).
            from model.cloth_masker import AutoMasker

            self.automasker = AutoMasker(
                densepose_ckpt=os.path.join(repo_path, "DensePose"),
                schp_ckpt=os.path.join(repo_path, "SCHP"),
                device=device,
            )

    def run(
        self,
        person_image: Image.Image,
        cloth_image: Image.Image,
        cloth_type: ClothType = "upper",
        person_mask: Optional[Image.Image] = None,  # bắt buộc nếu mask_free=False và không dùng automasker
        num_inference_steps: int = 40,   # 40 là điểm cân bằng tốt cho GPU 6GB; tăng 50-75 nếu VRAM/thời gian cho phép
        guidance_scale: float = 2.5,     # 2.0-3.5 thường cho kết quả tự nhiên nhất với CatVTON
        seed: Optional[int] = None,
        width: int = 576,                # giảm so với mặc định 768 gốc để vừa VRAM 6GB
        height: int = 768,               # giảm so với mặc định 1024 gốc
    ) -> Image.Image:
        """
        Trả về ảnh PIL kết quả người mặc đồ mới.
        """
        person_image = resize_and_crop(person_image.convert("RGB"), (width, height))
        cloth_image = resize_and_padding(cloth_image.convert("RGB"), (width, height))

        if self.mask_free:
            mask = None  # bản mask-free không cần mask đầu vào
        elif self.automasker is not None:
            mask = self.automasker(person_image, cloth_type)["mask"]
            mask = self.mask_processor.blur(mask, blur_factor=9)
        elif person_mask is not None:
            mask = resize_and_crop(person_mask.convert("L"), (width, height))
            mask = self.mask_processor.blur(mask, blur_factor=9)
        else:
            raise ValueError(
                "mask_free=False và không có automasker -> phải truyền person_mask thủ công"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Dọn cache trước khi generate để tối đa VRAM khả dụng cho bước diffusion
        if self.device == "cuda":
            torch.cuda.empty_cache()

        pipeline_kwargs = dict(
            image=person_image,
            condition_image=cloth_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if mask is not None:
            pipeline_kwargs["mask"] = mask

        result = self.pipeline(**pipeline_kwargs)[0]
        return result


# ---- Test nhanh từ CLI ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True, help="Đường dẫn ảnh người")
    parser.add_argument("--cloth", required=True, help="Đường dẫn ảnh quần/áo")
    parser.add_argument("--cloth_type", default="upper", choices=["upper", "lower", "overall"])
    parser.add_argument("--out", default="result.png")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-quality", action="store_true",
                         help="Dùng bản CatVTON gốc + AutoMasker + res 768x1024 (cần GPU >=8GB VRAM)")
    args = parser.parse_args()

    engine = TryOnEngine(
        mask_free=not args.full_quality,
        low_vram=not args.full_quality,
    )
    run_kwargs = dict(
        person_image=Image.open(args.person),
        cloth_image=Image.open(args.cloth),
        cloth_type=args.cloth_type,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    if args.full_quality:
        run_kwargs.update(width=768, height=1024)
    result = engine.run(**run_kwargs)
    result.save(args.out)
    print(f"Đã lưu kết quả tại: {args.out}")