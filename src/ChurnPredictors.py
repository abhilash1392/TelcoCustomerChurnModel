from pydantic import BaseModel


class ChurnPredictor(BaseModel):
    SeniorCitizen     :   int
    tenure            :   int
    MultipleLines     :   str
    InternetService   :   str
    OnlineSecurity    :   str
    OnlineBackup      :   str
    DeviceProtection  :   str
    TechSupport       :   str
    Contract          :   str
    PaperlessBilling  :   str
    PaymentMethod     :   str
    MonthlyCharges    :   float
    TotalCharges      :   float
