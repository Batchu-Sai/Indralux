from pptx import Presentation
import os

def extract_clean_images_from_pptx(pptx_path, output_dir):
    """
    Extracts only the clean embedded image blobs from a PowerPoint file,
    skipping text boxes, annotations, or rendered slide previews.
    Saves each as its own image file in the output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    prs = Presentation(pptx_path)
    saved = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        for shape_idx, shape in enumerate(slide.shapes, start=1):
            if hasattr(shape, "image"):  # This ensures it's a picture, not a shape or textbox
                image = shape.image
                ext = image.ext
                name = f"slide{slide_idx:02d}_img{shape_idx:02d}.{ext}"
                path = os.path.join(output_dir, name)
                with open(path, "wb") as f:
                    f.write(image.blob)
                saved.append(name)

    return saved


