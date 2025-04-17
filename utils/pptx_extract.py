from pptx import Presentation
import os

def extract_clean_images_from_pptx(pptx_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    prs = Presentation(pptx_path)
    image_files = []

    for slide_idx, slide in enumerate(prs.slides):
        for shape_idx, shape in enumerate(slide.shapes):
            if shape.shape_type == 13 and hasattr(shape, "image"):
                image = shape.image
                ext = image.ext
                filename = f"slide{slide_idx+1:02d}_img{shape_idx+1:02d}.{ext}"
                out_path = os.path.join(output_dir, filename)
                with open(out_path, "wb") as f:
                    f.write(image.blob)
                image_files.append(filename)
    return sorted(image_files)
