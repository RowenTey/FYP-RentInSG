import traceback
from models.prediction import PredictionFormData
from utils.mlflow_model import model
from cashews import cache
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter()


@router.post("/")
async def predict(data: PredictionFormData):
    try:
        form_data = data.model_dump()
        print(form_data)
        result = await model.predict(form_data)
        print(result)

        return JSONResponse(
            content=result,
            status_code=200
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing form data: {e}")


@router.post("/stream")
@cache(ttl="1h", key="data:{data:hash(sha1)}")
async def predict_stream(data: PredictionFormData):
    try:
        return StreamingResponse(await model.predict_stream(data.model_dump()), media_type="application/json")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
