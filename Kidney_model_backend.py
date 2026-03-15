import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import sklearn
import pandas as pd 
import os 
import shap
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all websites (development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sklearn.set_config(transform_output="pandas")
model_path = r"C:\FrostByte Project\Kidney disease prediction ml model(RFC).joblib"
model_url = "https://drive.google.com/file/d/1gcYGygP-BHErbvzP7opd7Uxu4KuvBzxc/view?usp=drive_link"
if not os.path.exists(model_path):
    print("Downloading model.......")
    gdown.download(model_url , model_path , quiet = False)
model = joblib.load(model_path)
### All columns more user friendly:
### sg	Urine Specific Gravity
### bp	Blood Pressure (mmHg)
### al	Urine Albumin Level
### su	Urine Sugar Level
### rbc	Red Blood Cells in Urine [normal , abnormal]
### pc	Pus Cells in Urine[normal , abnormal]
### pcc	Pus Cell Clumps [notpresent , present]
### ba	Bacteria in Urine [notpresent , present]
### bgr	Blood Glucose Random
### bu	Blood Urea
### sc	Serum Creatinine
### sod	Sodium Level
### pot	Potassium Level
### hemo	Hemoglobin
### pcv	Packed Cell Volume
### wc	White Blood Cell Count
### rc	Red Blood Cell Count
### htn	Hypertension (High BP) [False , True]
### dm	Diabetes Mellitus [yes , no]
### cad	Coronary Artery Disease [yes , no]
### appet	Appetite [good , poor]
### pe	Pedal Edema (Leg Swelling) [False , True]
### ane	Anemia [False , True]
### classification	CKD Diagnosis
class InputData(BaseModel):
    Age:int 
    Urine_Specific_Gravity:float
    Blood_Pressure_mmHg:int
    Urine_Albumin_Level:float
    Urine_Sugar_Level:float
    Red_Blood_Cells_Urine:str
    Pus_Cells_Urine:str
    Pus_Cell_Clumps:str
    Bacteria_Urine:str
    Blood_Glucose_Random:float
    Blood_Urea:float
    Serum_Creatinine:float
    Sodium_Level:float
    Potassium_Level:float
    Hemoglobin:float
    Packed_Cell_Volume:float
    White_Blood_Cell_Count:float
    Red_Blood_Cell_Count:float
    Hypertension_High_BP:str
    Diabetes_Mellitus:str
    Coronary_Artery_Disease:str
    Appetite:str
    Pedal_Edema_Leg_Swelling:str
    Anemia:str
feature_map = {
"sg":	"Urine Specific Gravity" , 
 "bp":	"Blood Pressure (mmHg)" , 
 "al":	"Urine Albumin Level" , 
 "su":	"Urine Sugar Level" , 
"rbc":	"Red Blood Cells in Urine" , 
"pc": 	"Pus Cells in Urine" , 
 "pcc"	:"Pus Cell Clumps" , 
 "ba":	"Bacteria in Urine" , 
 "bgr":	"Blood Glucose Random" , 
 "bu":	"Blood Urea" , 
 "sc": "Serum Creatinine" , 
 "sod" : 	"Sodium Level" , 
 "pot":	"Potassium Level" , 
 "hemo":	"Hemoglobin" , 
 "pcv"	:"Packed Cell Volume" , 
 "wc" : "White Blood Cell Count" , 
  "rc" :	"Red Blood Cell Count" , 
 "htn"	:"Hypertension (High BP)" , 
 "dm":	"Diabetes Mellitus", 
 "cad":	"Coronary Artery Disease" , 
 "appet":	"Appetite" , 
 "pe": "Pedal Edema (Leg Swelling)" , 
 "ane": "Anemia"
}
def Names_split(name):
    return name.split("__")[-1]
def Shap_explainations(df):
    df = model[:-1].transform(df)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    shap_values = explainer(df).values[0 , : , 0]
    feature_names = model[:-1].get_feature_names_out()
    Shap_df =  pd.DataFrame({
        "feature":feature_names , 
        "contribution" : shap_values
    })
    Shap_df['Absolute contribution'] = Shap_df['contribution'].abs()
    Shap_df["feature"] = Shap_df["feature"].apply(Names_split)
    Shap_df['feature'] = Shap_df['feature'].map(lambda x: feature_map.get(x , x))
    top_values = Shap_df.sort_values(
        by = "Absolute contribution" , ascending = False
    ).head(3)
    top_values['impact'] = top_values['contribution'].apply(lambda x: "Increase Kidney disease" if x > 0 else "Decrease Kidney disease")
    return top_values.to_dict(orient = "records")
@app.post("/predict")
def predict(data : InputData):
    df = {
        "age":data.Age , "sg":data.Urine_Specific_Gravity , "bp":data.Blood_Pressure_mmHg , "al":data.Urine_Albumin_Level , "su":data.Urine_Sugar_Level , "rbc":data.Red_Blood_Cells_Urine , "pc":data.Pus_Cells_Urine , 
        "pcc":data.Pus_Cell_Clumps , "ba":data.Bacteria_Urine , "bgr":data.Blood_Glucose_Random , "bu": data.Blood_Urea , "sc":data.Serum_Creatinine , 
        "sod":data.Sodium_Level  , "pot":data.Potassium_Level , "hemo":data.Hemoglobin , "pcv":data.Packed_Cell_Volume , "wc":data.White_Blood_Cell_Count , 
        "rc":data.Red_Blood_Cell_Count , "htn":data.Hypertension_High_BP , "dm":data.Diabetes_Mellitus , "cad":data.Coronary_Artery_Disease , "appet":data.Appetite , "pe":data.Pedal_Edema_Leg_Swelling, "ane":data.Anemia
        }
    df = pd.DataFrame([df])
    sklearn.set_config(transform_output="pandas")
    Explain = Shap_explainations(df)
    reply = model.predict(df)[0]
    prediction = model.predict_proba(df)[0 , 0]
    return {"Chances of Kidney disease": f"{round(prediction*100 , 4)}%" , "Risk Level":"Low" if reply == 1 else "High" , "Explain":Explain}