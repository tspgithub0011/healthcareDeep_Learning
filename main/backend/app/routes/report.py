"""POST /api/report — generate and return PDF report."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.models.schemas import ReportRequest
from app.utils.pdf_report import generate_pdf_report, REPORTLAB_AVAILABLE
from app.utils.logger import logger

router = APIRouter()


@router.post("/api/report")
async def create_report(request: ReportRequest):
    """
    Generate a PDF report from prediction results.

    Accepts the full prediction response JSON + optional base64 image.
    Returns the PDF as a downloadable file.
    """
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="PDF report generation is not available. Install reportlab: pip install reportlab",
        )

    try:
        pdf_bytes = generate_pdf_report(
            prediction_data=request.prediction,
            image_base64=request.image,
        )

        logger.info("PDF report generated successfully")

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=healthcare-dl-report.pdf"
            },
        )

    except Exception as e:
        logger.error(f"PDF report generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF report.")
