from pptx import Presentation
import os
from PIL import Image
import io

def extract_clean_images_from_pptx(pptx_path, output_dir):
    """
    Extracts only clean embedded image blobs from a PowerPoint file,
    skipping text boxes, annotations, or rendered slide previews.
    Saves each image file in the output_dir as clean RGB PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)
    prs = Presentation(pptx_path)
    saved = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        for shape_idx, shape in enumerate(slide.shapes, start=1):
            if hasattr(shape, "image"):
                image = shape.image
                name = f"slide{slide_idx:02d}_img{shape_idx:02d}.png"
                path = os.path.join(output_dir, name)

                try:
                    img = Image.open(io.BytesIO(image.blob)).convert("RGB")
                    img.save(path, format="PNG")
                    saved.append(name)
                except Exception as e:
                    print(f"[Indralux] Failed to convert image from slide {slide_idx}: {e}")
    return saved

