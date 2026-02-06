from pydantic import BaseModel, Field
from enum import Enum


# ------------------ ENUMS ------------------

class GenderEnum(str, Enum):
    Male = "Male"
    Female = "Female"


class YesNoEnum(str, Enum):
    Yes = "Yes"
    No = "No"


class ContractEnum(str, Enum):
    Month_to_month = "Month-to-month"
    One_year = "One year"
    Two_year = "Two year"


class PaymentMethodEnum(str, Enum):
    Credit_card = "Credit card"
    Electronic_check = "Electronic check"
    Mailed_check = "Mailed check"


class InternetEnum(str, Enum):
    DSL = "DSL"
    Fiber = "Fiber optic"
    No = "No"


# ------------------ INPUT SCHEMA ------------------

class CustomerInput(BaseModel):
    # -------- Numeric Features --------
    tenure: float = Field(
        ...,
        ge=0,
        le=100,
        description="Number of months the customer has stayed with the company",
        example=12
    )

    MonthlyCharges: float = Field(
        ...,
        ge=0,
        le=200,
        description="Monthly amount charged to the customer",
        example=75.5
    )

    TotalCharges: float = Field(
        ...,
        ge=0,
        le=25000,
        description="Total amount charged to the customer over tenure",
        example=900.0
    )

    # -------- Demographic --------
    gender: GenderEnum = Field(
        ...,
        description="Customer gender",
        example="Male"
    )

    SeniorCitizen: YesNoEnum = Field(
        ...,
        description="Whether the customer is a senior citizen",
        example="No"
    )

    Partner: YesNoEnum = Field(
        ...,
        description="Whether the customer has a partner",
        example="Yes"
    )

    Dependents: YesNoEnum = Field(
        ...,
        description="Whether the customer has dependents",
        example="No"
    )

    # -------- Services --------
    PhoneService: YesNoEnum = Field(
        ...,
        description="Whether the customer has phone service",
        example="Yes"
    )

    MultipleLines: YesNoEnum = Field(
        ...,
        description="Whether the customer has multiple phone lines",
        example="No"
    )

    InternetService: InternetEnum = Field(
        ...,
        description="Type of internet service used by the customer",
        example="Fiber optic"
    )

    OnlineSecurity: YesNoEnum = Field(
        ...,
        description="Whether the customer has online security service",
        example="No"
    )

    OnlineBackup: YesNoEnum = Field(
        ...,
        description="Whether the customer has online backup service",
        example="Yes"
    )

    DeviceProtection: YesNoEnum = Field(
        ...,
        description="Whether the customer has device protection service",
        example="No"
    )

    TechSupport: YesNoEnum = Field(
        ...,
        description="Whether the customer has tech support service",
        example="No"
    )

    StreamingTV: YesNoEnum = Field(
        ...,
        description="Whether the customer has streaming TV service",
        example="Yes"
    )

    StreamingMovies: YesNoEnum = Field(
        ...,
        description="Whether the customer has streaming movies service",
        example="No"
    )

    # -------- Contract & Billing --------
    Contract: ContractEnum = Field(
        ...,
        description="Customer contract duration",
        example="Month-to-month"
    )

    PaperlessBilling: YesNoEnum = Field(
        ...,
        description="Whether the customer uses paperless billing",
        example="Yes"
    )

    PaymentMethod: PaymentMethodEnum = Field(
        ...,
        description="Payment method used by the customer",
        example="Electronic check"
    )
