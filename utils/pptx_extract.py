from pptx import Presentation
import os

def extract_images_from_pptx(pptx_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    prs = Presentation(pptx_path)
    saved_files = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        for shape_idx, shape in enumerate(slide.shapes, start=1):
            if shape.shape_type == 13 and hasattr(shape, "image"):
                image = shape.image
                ext = image.ext
                name = f"slide{slide_idx:02d}_img{shape_idx:02d}.{ext}"
                path = os.path.join(output_dir, name)
                with open(path, "wb") as f:
                    f.write(image.blob)
                saved_files.append(name)

    return saved_files  # ✅ This fixes the error!

