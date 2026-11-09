import pytest

from starlette.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


VALID_CUSTOMER = {
    "id": 19556,
    "Gender": "Female",
    "Customer Type": "Loyal Customer",
    "Age": 52,
    "Type of Travel": "Business travel",
    "Class": "Eco",
    "Flight Distance": 160,
    "Inflight wifi service": 5,
    "Departure/Arrival time convenient": 4,
    "Ease of Online booking": 3,
    "Gate location": 4,
    "Food and drink": 3,
    "Online boarding": 4,
    "Seat comfort": 3,
    "Inflight entertainment": 5,
    "On-board service": 5,
    "Leg room service": 5,
    "Baggage handling": 5,
    "Checkin service": 2,
    "Inflight service": 5,
    "Cleanliness": 5,
    "Departure Delay in Minutes": 50,
    "Arrival Delay in Minutes": 44.0,
}


SECOND_CUSTOMER = {
    "id": 90035,
    "Gender": "Male",
    "Customer Type": "disloyal Customer",
    "Age": 36,
    "Type of Travel": "Personal Travel",
    "Class": "Business",
    "Flight Distance": 2863,
    "Inflight wifi service": 1,
    "Departure/Arrival time convenient": 1,
    "Ease of Online booking": 3,
    "Gate location": 1,
    "Food and drink": 5,
    "Online boarding": 4,
    "Seat comfort": 5,
    "Inflight entertainment": 4,
    "On-board service": 4,
    "Leg room service": 4,
    "Baggage handling": 4,
    "Checkin service": 3,
    "Inflight service": 4,
    "Cleanliness": 5,
    "Departure Delay in Minutes": 0,
    "Arrival Delay in Minutes": 0.0,
}


# ---------------------------------------------------------
# 9.1 - Health Endpoint
# ---------------------------------------------------------

def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ---------------------------------------------------------
# 9.2 - Metadata Endpoint
# ---------------------------------------------------------

def test_metadata(client):
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Airline Passenger Satisfaction API"
    assert data["version"] == "1.0.0"

    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True

    assert data["model_name"] is not None
    assert data["model_uri"] is not None


# ---------------------------------------------------------
# 9.3 - Valid Single Prediction
# ---------------------------------------------------------

def test_valid_prediction(client):
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "satisfaction_probability" in data

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["satisfaction_probability"] <= 1.0


# ---------------------------------------------------------
# 9.4 - Invalid Single Prediction: Gender
# ---------------------------------------------------------

def test_invalid_prediction_gender(client):
    payload = VALID_CUSTOMER.copy()

    payload["Gender"] = "ABC"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "Gender"


# ---------------------------------------------------------
# 9.4 - Invalid Single Prediction: Tenure
# ---------------------------------------------------------

def test_invalid_prediction_tenure(client):
    payload = VALID_CUSTOMER.copy()

    payload["Age"] = -5

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "Age"


# ---------------------------------------------------------
# 9.4 - Missing Required Field
# ---------------------------------------------------------

def test_missing_prediction_field(client):
    payload = VALID_CUSTOMER.copy()

    del payload["Flight Distance"]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "Flight Distance"


# ---------------------------------------------------------
# 9.5 - Valid Batch Prediction
# ---------------------------------------------------------

def test_valid_batch_prediction(client):
    payload = {
        "customers": [
            VALID_CUSTOMER,
            SECOND_CUSTOMER,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2

    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert "prediction" in prediction
        assert "satisfaction_probability" in prediction

        assert prediction["prediction"] in [0, 1]

        assert 0.0 <= prediction["satisfaction_probability"] <= 1.0


# ---------------------------------------------------------
# 9.6 - Invalid Batch Prediction
# ---------------------------------------------------------

def test_invalid_batch_prediction(client):
    invalid_customer = SECOND_CUSTOMER.copy()

    invalid_customer["gender"] = "ABC"

    payload = {
        "customers": [
            VALID_CUSTOMER,
            invalid_customer,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "customers.1.Gender"


# ---------------------------------------------------------
# 9.7 - Multiple Batch Validation Errors
# ---------------------------------------------------------

def test_multiple_batch_validation_errors(client):
    first_customer = VALID_CUSTOMER.copy()
    second_customer = SECOND_CUSTOMER.copy()

    first_customer["gender"] = "ABC"
    second_customer["Age"] = -5

    payload = {
        "customers": [
            first_customer,
            second_customer,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 2

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "customers.0.Gender" in fields
    assert "customers.1.Age" in fields


# ---------------------------------------------------------
# 9.8 - Empty Batch
# ---------------------------------------------------------

def test_empty_batch(client):
    payload = {
        "customers": []
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "customers"


# ---------------------------------------------------------
# 9.9 - Invalid Datatype
# ---------------------------------------------------------

def test_invalid_datatype(client):
    payload = VALID_CUSTOMER.copy()

    payload["Age"] = "ten"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "Age"


# ---------------------------------------------------------
# 9.10 - Extreme Valid Numeric Values
# ---------------------------------------------------------

def test_extreme_numeric_values(client):
    payload = VALID_CUSTOMER.copy()

    payload["Age"] = 100
    payload["Flight Distance"] = 10000.0
    payload["Arrival Delay in Minutes"] = 1000000.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]

    assert 0.0 <= data["satisfaction_probability"] <= 1.0

# ---------------------------------------------------------
# Stage 10.1 - Null / Corrupted Input
# ---------------------------------------------------------

def test_null_value(client):
    payload = VALID_CUSTOMER.copy()

    payload["Gender"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "Gender"

# =========================================================
# STAGE 10 - INVALID / EXTREME / CORRUPTED INPUT TESTING
# =========================================================


# ---------------------------------------------------------
# 10.2 - Malformed JSON
# ---------------------------------------------------------

def test_malformed_json(client):
    malformed_json = """
    {
        "id": "TEST001",
        "Gender": "Male"
    """

    response = client.post(
        "/predict",
        content=malformed_json,
        headers={
            "Content-Type": "application/json"
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.3 - Completely Wrong JSON Structure
# ---------------------------------------------------------

def test_wrong_json_structure(client):
    payload = [
        "this",
        "should",
        "be",
        "an",
        "object"
    ]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.4 - Null Numeric Value
# ---------------------------------------------------------

def test_null_numeric_value(client):
    payload = VALID_CUSTOMER.copy()

    payload["Flight Distance"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "Flight Distance"


# ---------------------------------------------------------
# 10.5 - Null Required String Value
# ---------------------------------------------------------

def test_null_total_charges(client):
    payload = VALID_CUSTOMER.copy()

    payload["Arrival Delay in Minutes"] = None

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) == 1

    assert data["details"][0]["field"] == "Arrival Delay in Minutes"


# ---------------------------------------------------------
# 10.6 - Wrong Datatype for Multiple Fields
# ---------------------------------------------------------

def test_multiple_invalid_datatypes(client):
    payload = VALID_CUSTOMER.copy()

    payload["Age"] = "ten years"
    payload["Flight Distance"] = "fifty"
    payload["SeniorCitizen"] = "old"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 2

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "Age" in fields
    assert "Flight Distance" in fields


# ---------------------------------------------------------
# 10.7 - Invalid Batch Structure
# ---------------------------------------------------------

def test_invalid_batch_structure(client):
    payload = {
        "customers": VALID_CUSTOMER
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.8 - Null Batch Customers
# ---------------------------------------------------------

def test_null_batch_customers(client):
    payload = {
        "customers": None
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.9 - Batch Containing Null Customer
# ---------------------------------------------------------

def test_batch_with_null_customer(client):
    payload = {
        "customers": [
            VALID_CUSTOMER,
            None,
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    assert len(data["details"]) >= 1


# ---------------------------------------------------------
# 10.10 - Extreme Negative Numeric Values
# ---------------------------------------------------------

def test_extreme_negative_values(client):
    payload = VALID_CUSTOMER.copy()

    payload["Age"] = -1000000
    payload["Flight Distance"] = -999999.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    fields = [
        error["field"]
        for error in data["details"]
    ]

    assert "Age" in fields
    assert "Flight Distance" in fields


# ---------------------------------------------------------
# 10.11 - Very Large Valid Batch
# ---------------------------------------------------------

def test_large_batch(client):
    customers = []

    for i in range(20):
        customer = VALID_CUSTOMER.copy()

        customer["id"] = 1000 + i

        customers.append(customer)

    payload = {
        "customers": customers
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 20

    assert len(data["predictions"]) == 20

    for prediction in data["predictions"]:
        assert prediction["prediction"] in [0, 1]

        assert (
            0.0
            <= prediction["satisfaction_probability"]
            <= 1.0
        )