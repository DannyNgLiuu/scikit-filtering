from pydantic import BaseModel
from typing import Union

class PreferenceWeights(BaseModel):
    Region: Union[bool, float] = False
    Game_Genre: Union[bool, float] = False
    Preferred_Game_Mode: Union[bool, float] = False
    Platform: Union[bool, float] = False
    Playstyle_Tags: Union[bool, float] = False
    Skill_Level: Union[bool, float] = False
    Reputation: Union[bool, float] = 0
    Reports: Union[bool, float] = 0
    Friend_List_Overlap: Union[bool, float] = 0
    Age: Union[bool, float] = 0
