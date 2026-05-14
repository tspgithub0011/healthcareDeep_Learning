"""Image upload validation — file type, size, corruption, dimensions."""
from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import settings
from app.utils.logger import logger


async def validate_upload(file: UploadFile) -> Image.Image:
    """
    Validate an uploaded file and return a PIL Image.

    Checks:
    1. MIME type is allowed (JPEG, PNG, WEBP)
    2. File size ≤ MAX_FILE_SIZE_MB
    3. File is not corrupt (PIL can open it)
    4. Dimensions are at least 32×32
    5. Convert to RGB mode if needed

    Returns:
        PIL.Image.Image in RGB mode

    Raises:
        HTTPException 413 for oversized files
        HTTPException 422 for invalid files
    """
    # 1. Check MIME type
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        logger.warning(f"Rejected file: invalid type '{file.content_type}'")
        raise HTTPException(
            status_code=422,
            detail=f"Invalid image: file is not a supported format. Accepted: JPEG, PNG, WEBP.",
        )

    # 2. Read file contents and check size
    contents = await file.read()

    if len(contents) > settings.MAX_FILE_SIZE_BYTES:
        size_mb = len(contents) / (1024 * 1024)
        logger.warning(f"Rejected file: too large ({size_mb:.1f} MB)")
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum size is {settings.MAX_FILE_SIZE_MB} MB.",
        )

    # 3. Try to open with PIL (corruption check)
    try:
        image = Image.open(BytesIO(contents))
        image.load()  # Force full decode to catch truncated images
    except Exception:
        logger.warning("Rejected file: corrupt or unreadable image")
        raise HTTPException(
            status_code=422,
            detail="Invalid image: file appears to be corrupt or unreadable.",
        )

    # 4. Check minimum dimensions
    width, height = image.size
    if width < settings.MIN_IMAGE_DIMENSION or height < settings.MIN_IMAGE_DIMENSION:
        logger.warning(f"Rejected file: too small ({width}x{height})")
        raise HTTPException(
            status_code=422,
            detail=f"Image too small ({width}×{height}). Minimum dimension is {settings.MIN_IMAGE_DIMENSION}×{settings.MIN_IMAGE_DIMENSION} pixels.",
        )

    # 5. Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    logger.info(f"Validated image: {file.filename} ({width}x{height}, {len(contents)/1024:.0f} KB)")
    return image
