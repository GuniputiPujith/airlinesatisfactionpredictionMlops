from typing import Literal
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    id: int
    Gender: Literal["Male", "Female"]
    Customer_Type: Literal["Loyal Customer", "disloyal Customer"] = Field(alias="Customer Type")
    Age: int = Field(ge=0, le=120)
    Type_of_Travel: Literal["Business travel", "Personal Travel"] = Field(alias="Type of Travel")
    Class: Literal["Business", "Eco", "Eco Plus"]
    Flight_Distance: int = Field(alias="Flight Distance", ge=0)
    Inflight_wifi_service: int = Field(alias="Inflight wifi service", ge=0, le=5)
    Departure_Arrival_time_convenient: int = Field(alias="Departure/Arrival time convenient", ge=0, le=5)
    Ease_of_Online_booking: int = Field(alias="Ease of Online booking", ge=0, le=5)
    Gate_location: int = Field(alias="Gate location", ge=0, le=5)
    Food_and_drink: int = Field(alias="Food and drink", ge=0, le=5)
    Online_boarding: int = Field(alias="Online boarding", ge=0, le=5)
    Seat_comfort: int = Field(alias="Seat comfort", ge=0, le=5)
    Inflight_entertainment: int = Field(alias="Inflight entertainment", ge=0, le=5)
    On_board_service: int = Field(alias="On-board service", ge=0, le=5)
    Leg_room_service: int = Field(alias="Leg room service", ge=0, le=5)
    Baggage_handling: int = Field(alias="Baggage handling", ge=0, le=5)
    Checkin_service: int = Field(alias="Checkin service", ge=0, le=5)
    Inflight_service: int = Field(alias="Inflight service", ge=0, le=5)
    Cleanliness: int = Field(ge=0, le=5)
    Departure_Delay_in_Minutes: int = Field(alias="Departure Delay in Minutes", ge=0)
    Arrival_Delay_in_Minutes: float = Field(alias="Arrival Delay in Minutes", ge=0)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: int
    satisfaction_probability: float


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
