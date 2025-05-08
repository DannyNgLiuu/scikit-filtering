from fastapi import APIRouter, HTTPException
from schemas.weights import PreferenceWeights
from config.database import engine
from model import recommender
from model.preprocessing import build_preprocessor
import pandas as pd

router = APIRouter()


@router.on_event("startup")
async def startup():
    preprocessor = build_preprocessor()
    recommender.init_model(preprocessor)


@router.get("/recommendations/{player_id}")
async def get_recommendations(player_id: int, num_recommendations: int = 5):
    df = pd.read_sql_query("SELECT * FROM user_profiles", engine)

    if player_id not in df['Player ID'].values:
        raise HTTPException(status_code=404, detail="Player ID not found")

    player_index = df[df['Player ID'] == player_id].index[0]

    return recommender.get_recommendations_ml(
        player_index, num_recommendations
    ).to_dict(orient="records")


@router.post("/recommendations/{player_id}/custom")
async def get_custom(player_id: int, preferences: PreferenceWeights, num_recommendations: int = 5):
    df = pd.read_sql_query("SELECT * FROM user_profiles", engine)

    if player_id not in df['Player ID'].values:
        raise HTTPException(status_code=404, detail="Player ID not found")

    player_index = df[df['Player ID'] == player_id].index[0]

    pref_dict = preferences.dict()
    for k in pref_dict:
        if isinstance(pref_dict[k], bool):
            pref_dict[k] = 1 if pref_dict[k] else 0

    pref_dict = {k.replace('_', ' '): v for k, v in pref_dict.items()}
    weighted = recommender.apply_custom_weights(pref_dict)

    return recommender.get_recommendations_ml(
        player_index, num_recommendations, weighted
    ).to_dict(orient="records")


@router.post("/refresh")
async def refresh_model():
    recommender.refresh_model()
    return {"message": "Model refreshed with latest users"}
